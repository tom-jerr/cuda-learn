#include <cstdio>
#include <cstdlib>
#include <vector>

#include "multi_stage_config.cuh"

// 不运行GPU指令，用真实Config核对文档中的tile移动、源/目的地址差与retile槽位。
// 先读verify_tile_steps：它直接观察“下一块”相对“这一块”的地址差。
// 后面的逐元素坐标检查用于覆盖所有lane，公式细节对应文档折叠附录。
namespace half_gemm {
namespace {
void verify(bool ok, char const* message) {
  if (!ok) {
    std::fprintf(stderr, "LAYOUT FAIL: %s\n", message);
    std::exit(1);
  }
}
template <class T>
void show(char const* name, T const& value) {
  std::printf("%s: ", name);
  print(value);
  std::puts("");
}
// 从“相邻 tile 起点”测量 view 的实际步长。数值单位均为 half。
// 不用 shape 猜存储顺序：partition 保留原地址，fragment 才拥有本线程存储。
void verify_tile_steps() {
  using Cfg = Config;
  std::vector<half_t> global_storage(128 * 256), shared_storage(cosize(Cfg::SmemLayoutA{}));
  auto full = make_tensor(global_storage.data(), Layout<Shape<_128, _256>, Stride<_256, _1>>{});
  auto ga = local_tile(full, Shape<_128, _32>{}, make_coord(0, _));
  auto gc = local_tile(full, Shape<_128, _128>{}, make_coord(0, 0));
  auto sa = make_tensor(make_smem_ptr(shared_storage.data()), Cfg::SmemLayoutA{});
  Cfg::MMA mma;
  Cfg::G2SCopyA g2s;
  [[maybe_unused]] auto la = make_tiled_copy_A(Cfg::S2RCopyAtomA{}, mma);
  [[maybe_unused]] auto lb = make_tiled_copy_B(Cfg::S2RCopyAtomB{}, mma);
  int positive_q = 0, negative_q = 0;
  for (int tid = 0; tid < 128; ++tid) {
    auto gs = g2s.get_slice(tid).partition_S(ga);
    auto sd = g2s.get_slice(tid).partition_D(sa);
    // G2S 的基本 tile=32x32；向下跨32行，global和shared的行宽不同。
    for (int r = 0; r < 4; ++r)
      for (int v = 0; v < 8; ++v) {
        for (int kt = 0; kt < 8; ++kt)
          verify(&gs(v, r, 0, kt) - &gs(0, 0, 0, 0) == v + 8192 * r + 32 * kt,
                 "G2S global: value +1, row tile +8192, K tile +32");
        for (int st = 0; st < 3; ++st)
          verify(&sd(v, r, 0, st) - &sd(0, 0, 0, 0) == v + 1024 * r + 4096 * st,
                 "G2S shared: value +1, row tile +1024, stage +4096");
      }
    auto as = la.get_slice(tid).partition_S(sa);
    auto bs = lb.get_slice(tid).partition_S(sa);
    // q 从0到1表示跨K16，swizzle后不一定是正向16个元素。
    int aq = int(&as(0, 0, 1, 0) - &as(0, 0, 0, 0));
    int bq = int(&bs(0, 0, 1, 0) - &bs(0, 0, 0, 0));
    verify(aq == 16 || aq == -16, "A S2R q delta is signed 16");
    verify(bq == 16 || bq == -16, "B S2R q delta is signed 16");
    if (aq > 0)
      ++positive_q;
    else
      ++negative_q;
    for (int st = 0; st < 3; ++st)
      for (int r = 0; r < 4; ++r)
        for (int q = 0; q < 2; ++q)
          for (int v = 0; v < 8; ++v) {
            verify(&as(v, r, q, st) - &as(0, 0, 0, 0) == v + 1024 * r + aq * q + 4096 * st,
                   "A S2R shared tile steps");
            verify(&bs(v, r, q, st) - &bs(0, 0, 0, 0) == v + 1024 * r + bq * q + 4096 * st,
                   "B S2R shared tile steps");
          }
    auto th = mma.get_slice(tid);
    auto pa = th.partition_A(ga(_, _, 0));
    auto pb = th.partition_B(ga(_, _, 0));
    auto pc = th.partition_C(gc);
    auto ra = th.partition_fragment_A(ga(_, _, 0));
    auto rb = th.partition_fragment_B(ga(_, _, 0));
    auto rc = th.partition_fragment_C(gc);
    // 同一次逻辑移动，在global view和本线程fragment中得到不同的地址差。
    verify(&pa(0, 1, 0) - &pa(0, 0, 0) == 8192 && &ra(0, 1, 0) - &ra(0, 0, 0) == 16,
           "A row32: global +8192, registers +16");
    verify(&pa(0, 0, 1) - &pa(0, 0, 0) == 16 && &ra(0, 0, 1) - &ra(0, 0, 0) == 8,
           "A K16: global +16, registers +8");
    verify(&pb(0, 1, 0) - &pb(0, 0, 0) == 4096 && &rb(0, 1, 0) - &rb(0, 0, 0) == 8,
           "B N16: global +4096, registers +8");
    verify(&pb(0, 0, 1) - &pb(0, 0, 0) == 16 && &rb(0, 0, 1) - &rb(0, 0, 0) == 4,
           "B K16: global +16, registers +4");
    verify(&pc(0, 1, 0) - &pc(0, 0, 0) == 8192 && &rc(0, 1, 0) - &rc(0, 0, 0) == 4,
           "C M32: global +8192, registers +4");
    verify(&pc(0, 0, 1) - &pc(0, 0, 0) == 16 && &rc(0, 0, 1) - &rc(0, 0, 0) == 16,
           "C N16: global +16, registers +16");
    if (tid == 0) {
      show("G2S global source", gs.layout());
      show("G2S shared destination", sd.layout());
      show("MMA partition_A(global)", pa.layout());
      show("MMA partition_B(global)", pb.layout());
      show("MMA partition_C(global)", pc.layout());
      show("S2R A register retile", la.get_slice(tid).retile_D(ra).layout());
      show("S2R B register retile", lb.get_slice(tid).retile_D(rb).layout());
    }
  }
  verify(positive_q == 64 && negative_q == 64, "both signs of shared K16 step are covered");
  std::puts("PASS tile steps: global/shared/register deltas; S2R K16 has 64 +16 and 64 -16 lanes");
}

}  // namespace
void verify_layouts() {
  verify_tile_steps();
  using Cfg = Config;
  auto coord_ab = make_identity_tensor(Shape<_128, _32>{});
  auto coord_stage = make_identity_tensor(Shape<_128, _32, _3>{});
  auto coord_c = make_identity_tensor(Shape<_128, _128>{});
  auto coord_scratch = make_identity_tensor(Shape<_32, _32, _2>{});
  std::vector<half_t> storage(32768);
  // 与 kernel 相同，fragment 从 global row-stride=256 的视图创建。
  auto ab = make_tensor(storage.data(), Layout<Shape<_128, _32>, Stride<_256, _1>>{});
  auto gc = make_tensor(storage.data(), Layout<Shape<_128, _128>, Stride<_256, _1>>{});
  Cfg::MMA mma;
  Cfg::G2SCopyA g2s;
  Cfg::S2GCopyC s2g;
  [[maybe_unused]] auto la = make_tiled_copy_A(Cfg::S2RCopyAtomA{}, mma);
  [[maybe_unused]] auto lb = make_tiled_copy_B(Cfg::S2RCopyAtomB{}, mma);
  auto r2s = make_tiled_copy_C(Cfg::R2SCopyAtomC{}, mma);
  std::vector<int> a_owners(4096), b_owners(4096), c_owners(16384), writers(16384), readers(16384);
  std::vector<int> slots(12288);
  for (int st = 0; st < 3; ++st)
    for (int r = 0; r < 128; ++r)
      for (int c = 0; c < 32; ++c) {
        int x = 4096 * st + 32 * r + c;
        int p = int(Cfg::SmemLayoutA{}(r, c, st));
        verify(p == (x ^ ((x & 0x1c0) >> 3)), "input swizzle XOR");
        verify(p == 4096 * st + 32 * (r ^ ((r >> 3) & 1)) + (c ^ (8 * ((r / 2) % 4))),
               "input row/col formula");
        verify(p >= st * 4096 && p < (st + 1) * 4096, "input stage isolation");
        ++slots[p];
        if (c % 8 == 0) verify((2 * p) % 16 == 0, "16-byte input row alignment");
      }
  for (int count : slots) verify(count == 1, "input physical coverage");
  std::vector<int> cslots(2048);
  for (int b = 0; b < 2; ++b)
    for (int r = 0; r < 32; ++r)
      for (int c = 0; c < 32; ++c) {
        int p = int(Cfg::SmemLayoutC{}(r, c, b));
        verify(p == 1024 * b + 32 * r + (c ^ (8 * ((r / 2) % 4))), "output swizzle");
        ++cslots[p];
        if (c % 8 == 0) verify(2 * p % 16 == 0, "output vector alignment");
      }
  for (int count : cslots) verify(count == 1, "output scratch coverage");
  for (int tid = 0; tid < 128; ++tid) {
    int w = tid / 32, l = tid % 32, wm = w % 2, wn = w / 2, g = l / 4, t = l % 4;
    auto th = mma.get_slice(tid);
    auto a = th.partition_A(coord_ab);
    auto b = th.partition_B(coord_ab);
    auto c = th.partition_C(coord_c);
    // 同样的tile在本线程存储中更紧凑：A跨M32/K16为16/8槽，B跨N16/K16为8/4槽。
    auto ra = th.partition_fragment_A(ab);
    auto rb = th.partition_fragment_B(ab);
    auto rc = th.partition_fragment_C(gc);
    auto ac = la.get_slice(tid).retile_D(ra);
    auto bc = lb.get_slice(tid).retile_D(rb);
    auto as = la.get_slice(tid).partition_S(coord_stage);
    auto bs = lb.get_slice(tid).partition_S(coord_stage);
    auto load = g2s.get_slice(tid).partition_D(coord_stage);
    auto scatter = group_modes<1, 3>(r2s.get_slice(tid).retile_S(rc));
    auto sc = r2s.get_slice(tid).partition_D(coord_scratch);
    auto out = group_modes<1, 3>(s2g.get_slice(tid).partition_D(coord_c));
    auto sr = s2g.get_slice(tid).partition_S(coord_scratch);
    verify(size(ra) == 64 && size(rb) == 64 && size(rc) == 128, "fragment element counts");
    for (int i = 0; i < 4; ++i)
      for (int q = 0; q < 2; ++q)
        for (int v = 0; v < 8; ++v) {
          int m = 16 * wm + g + 8 * ((v / 2) % 2) + 32 * i,
              k = 2 * t + v % 2 + 8 * (v / 4) + 16 * q;
          verify(a(v, i, q) == make_coord(m, k), "A lane coordinate");
          ++a_owners[m * 32 + k];
          verify(int(ra.layout()(v, i, q)) == v + 16 * i + 8 * q, "A register offset");
          verify(&ac(v, i, q) == &ra(v, i, q), "A retile alias");
          verify(&bc(v, i, q) == &rb(v % 4, 2 * i + v / 4, q), "B retile folds N pair");
        }
    for (int j = 0; j < 8; ++j)
      for (int q = 0; q < 2; ++q)
        for (int v = 0; v < 4; ++v) {
          int n = 8 * wn + g + 16 * j, k = 2 * t + v % 2 + 8 * (v / 2) + 16 * q;
          verify(b(v, j, q) == make_coord(n, k), "B lane coordinate");
          ++b_owners[n * 32 + k];
          verify(int(rb.layout()(v, j, q)) == v + 8 * j + 4 * q, "B register offset");
        }
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j)
        for (int v = 0; v < 4; ++v) {
          int m = 16 * wm + g + 8 * (v / 2) + 32 * i, n = 8 * wn + 2 * t + v % 2 + 16 * j;
          verify(c(v, i, j) == make_coord(m, n), "C lane coordinate");
          ++c_owners[m * 128 + n];
          verify(int(rc.layout()(v, i, j)) == v + 4 * i + 16 * j, "C register offset");
        }
    for (int st = 0; st < 3; ++st)
      for (int i = 0; i < 4; ++i) {
        for (int v = 0; v < 8; ++v)
          verify(load(v, i, 0, st) == make_coord(tid / 4 + 32 * i, 8 * (tid % 4) + v, st),
                 "G2S coordinates");
        for (int q = 0; q < 2; ++q) {
          verify(
              as(0, i, q, st) == make_coord(16 * wm + l % 16 + 32 * i, 8 * (l / 16) + 16 * q, st),
              "A ldmatrix row");
          verify(bs(0, i, q, st) == make_coord(8 * wn + l % 8 + 16 * (l / 16) + 32 * i,
                                               8 * ((l / 8) % 2) + 16 * q, st),
                 "B ldmatrix row");
        }
      }
    // 输出按16个32x32宏块遍历。R2S继承MMA分工，S2G换成连续8half向量。
    // 两端共同的p只表示同一宏块；它们的槽位/地址步长不必相同。
    for (int p = 0; p < 16; ++p)
      for (int u = 0; u < 8; ++u) {
        int i = p % 4, j = 2 * (p / 4) + u / 4, v = u % 4;
        verify(&scatter(u, p) == &rc(v, i, j), "R2S grouped alias");
        auto local = sc(u, 0, 0, p % 2);
        int m = int(get<0>(local)) + 32 * (p % 4), n = int(get<1>(local)) + 32 * (p / 4);
        verify(c(v, i, j) == make_coord(m, n), "R2S logical coordinate");
        ++writers[m * 128 + n];
        auto global = out(u, p);
        auto local_read = sr(u, 0, 0, p % 2);
        int om = tid / 4 + 32 * (p % 4), on = 8 * (tid % 4) + u + 32 * (p / 4);
        verify(global == make_coord(om, on), "S2G global coordinate");
        verify(local_read == make_coord(tid / 4, 8 * (tid % 4) + u, p % 2),
               "S2G scratch coordinate");
        ++readers[om * 128 + on];
      }
  }
  for (int i = 0; i < 4096; ++i) verify(a_owners[i] == 2 && b_owners[i] == 2, "A/B duplication");
  for (int i = 0; i < 16384; ++i)
    verify(c_owners[i] == 1 && writers[i] == 1 && readers[i] == 1, "C ownership bijection");
  show("MMA", mma);
  show("SmemLayoutA", Cfg::SmemLayoutA{});
  show("SmemLayoutC", Cfg::SmemLayoutC{});
  auto th = mma.get_slice(0);
  show("tCrA", th.partition_fragment_A(ab).layout());
  show("tCrB", th.partition_fragment_B(ab).layout());
  show("tCrD", th.partition_fragment_C(gc).layout());
  std::puts(
      "PASS layout: 128 threads, all stages, coordinates, offsets, retile and epilogue batches");
}
}  // namespace half_gemm
