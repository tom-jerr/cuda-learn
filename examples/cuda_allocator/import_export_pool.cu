// cuda_mempool_ipc_minimal.cu
// Linux only.
//
// Build:
//   nvcc -std=c++17 -O2 cuda_mempool_ipc_minimal.cu -o cuda_mempool_ipc_minimal
//
// Run:
//   ./cuda_mempool_ipc_minimal
//
// Expected:
//   [parent] verification: PASS

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#define CUDA(call)                                                             \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static void xwrite(int fd, const void *p, size_t n) {
  const char *s = (const char *)p;
  while (n) {
    ssize_t k = write(fd, s, n);
    if (k < 0 && errno == EINTR)
      continue;
    if (k <= 0) {
      perror("write");
      exit(1);
    }
    s += k;
    n -= (size_t)k;
  }
}

static void xread(int fd, void *p, size_t n) {
  char *s = (char *)p;
  while (n) {
    ssize_t k = read(fd, s, n);
    if (k < 0 && errno == EINTR)
      continue;
    if (k <= 0) {
      perror("read");
      exit(1);
    }
    s += k;
    n -= (size_t)k;
  }
}

// POSIX FD must be transferred with SCM_RIGHTS.
// Sending the integer fd value itself is not enough.
static void send_fd(int sock, int fd) {
  char byte = 'F';
  iovec iov{&byte, 1};

  char control[CMSG_SPACE(sizeof(int))]{};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);

  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));

  if (sendmsg(sock, &msg, 0) < 0) {
    perror("sendmsg");
    exit(1);
  }
}

static int recv_fd(int sock) {
  char byte;
  iovec iov{&byte, 1};

  char control[CMSG_SPACE(sizeof(int))]{};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);

  if (recvmsg(sock, &msg, 0) < 0) {
    perror("recvmsg");
    exit(1);
  }

  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS) {
    fprintf(stderr, "failed to receive pool fd\n");
    exit(1);
  }

  int fd;
  memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
  return fd;
}

__global__ void write_kernel(float *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = 100.0f + i;
}

// ------------------------------------------------------------
// Child: importer
// ------------------------------------------------------------
static void importer(int sock) {
  CUDA(cudaSetDevice(0));

  cudaStream_t stream;
  CUDA(cudaStreamCreate(&stream));

  // 1. import pool
  int pool_fd = recv_fd(sock);

  cudaMemPool_t pool;
  CUDA(cudaMemPoolImportFromShareableHandle(
      &pool, (void *)(intptr_t)pool_fd, cudaMemHandleTypePosixFileDescriptor,
      0));

  // 2. import one allocation from that pool
  cudaMemPoolPtrExportData data{};
  xread(sock, &data, sizeof(data));

  float *ptr = nullptr;
  CUDA(cudaMemPoolImportPointer((void **)&ptr, pool, &data));

  printf("[child ] imported ptr = %p\n", (void *)ptr);

  // 3. wait until exporter's async allocation is valid
  cudaIpcEventHandle_t ready_handle{};
  xread(sock, &ready_handle, sizeof(ready_handle));

  cudaEvent_t ready;
  CUDA(cudaIpcOpenEventHandle(&ready, ready_handle));
  CUDA(cudaStreamWaitEvent(stream, ready, 0));

  // use shared memory
  constexpr int N = 8;
  write_kernel<<<1, 32, 0, stream>>>(ptr, N);
  CUDA(cudaGetLastError());

  // 4. importer frees FIRST
  CUDA(cudaFreeAsync(ptr, stream));

  // DONE is after the importer's free in stream order
  cudaEvent_t done;
  CUDA(cudaEventCreateWithFlags(&done, cudaEventInterprocess |
                                           cudaEventDisableTiming));
  CUDA(cudaEventRecord(done, stream));

  cudaIpcEventHandle_t done_handle{};
  CUDA(cudaIpcGetEventHandle(&done_handle, done));
  xwrite(sock, &done_handle, sizeof(done_handle));

  // Keep child's CUDA objects alive until parent has consumed DONE.
  char ack;
  xread(sock, &ack, 1);

  CUDA(cudaStreamSynchronize(stream));
  CUDA(cudaEventDestroy(ready));
  CUDA(cudaEventDestroy(done));
  CUDA(cudaMemPoolDestroy(pool));
  CUDA(cudaStreamDestroy(stream));

  close(sock);
}

// ------------------------------------------------------------
// Parent: exporter
// ------------------------------------------------------------
static void exporter(int sock) {
  CUDA(cudaSetDevice(0));

  cudaStream_t stream;
  CUDA(cudaStreamCreate(&stream));

  // 1. create exportable pool
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = 0;
  props.handleTypes = cudaMemHandleTypePosixFileDescriptor;

  cudaMemPool_t pool;
  CUDA(cudaMemPoolCreate(&pool, &props));

  int pool_fd;
  CUDA(cudaMemPoolExportToShareableHandle(
      &pool_fd, pool, cudaMemHandleTypePosixFileDescriptor, 0));

  send_fd(sock, pool_fd);
  close(pool_fd);

  // 2. async allocation from this explicit pool
  constexpr int N = 8;
  float *ptr = nullptr;

  CUDA(cudaMallocFromPoolAsync((void **)&ptr, N * sizeof(float), pool, stream));

  printf("[parent] exporter ptr = %p\n", (void *)ptr);

  // export allocation identity
  cudaMemPoolPtrExportData data{};
  CUDA(cudaMemPoolExportPointer(&data, ptr));
  xwrite(sock, &data, sizeof(data));

  // READY: mallocAsync happens-before child access
  cudaEvent_t ready;
  CUDA(cudaEventCreateWithFlags(&ready, cudaEventInterprocess |
                                            cudaEventDisableTiming));
  CUDA(cudaEventRecord(ready, stream));

  cudaIpcEventHandle_t ready_handle{};
  CUDA(cudaIpcGetEventHandle(&ready_handle, ready));
  xwrite(sock, &ready_handle, sizeof(ready_handle));

  // 3. wait for child's imported-pointer free
  cudaIpcEventHandle_t done_handle{};
  xread(sock, &done_handle, sizeof(done_handle));

  cudaEvent_t done;
  CUDA(cudaIpcOpenEventHandle(&done, done_handle));
  CUDA(cudaStreamWaitEvent(stream, done, 0));

  // Child's imported mapping is freed, but exporter still owns allocation.
  float host[N]{};
  CUDA(
      cudaMemcpyAsync(host, ptr, sizeof(host), cudaMemcpyDeviceToHost, stream));

  // 4. exporter frees LAST
  CUDA(cudaFreeAsync(ptr, stream));
  CUDA(cudaStreamSynchronize(stream));

  printf("[parent] values:");
  for (float x : host)
    printf(" %.0f", x);
  printf("\n");

  bool ok = true;
  for (int i = 0; i < N; ++i)
    ok &= (host[i] == 100.0f + i);

  printf("[parent] verification: %s\n", ok ? "PASS" : "FAIL");

  char ack = 'K';
  xwrite(sock, &ack, 1);

  CUDA(cudaEventDestroy(done));
  CUDA(cudaEventDestroy(ready));
  CUDA(cudaMemPoolDestroy(pool));
  CUDA(cudaStreamDestroy(stream));

  close(sock);
}

int main() {

  // Fork BEFORE either process initializes CUDA.
  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
    perror("socketpair");
    return 1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 1;
  }

  if (pid == 0) {
    close(sv[0]);
    importer(sv[1]);
    return 0;
  }

  close(sv[1]);
  exporter(sv[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
