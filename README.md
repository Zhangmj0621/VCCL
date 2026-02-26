VCCL is a collective communication library for GPUs. It provides communication primitives such as **all-reduce, all-gather, reduce, broadcast, reduce-scatter**, and general **send/recv**. It is compatible with **PCIe, NVLink, and NVSwitch**, and supports **cross-node communication via InfiniBand Verbs or TCP/IP sockets**. It can be used in single-node/multi-node, multi-process (e.g., MPI), or single-process applications.


---
## Build from Source

> **Note**: Currently, only source builds are supported.

```bash
$ git clone ...
$ cd VCCL
$ make -j src.build
```

If CUDA is not installed under /usr/local/cuda:

```bash
$ make src.build CUDA_HOME=<path to cuda install>
```

Build artifacts are placed in the build/ directory (can be customized via BUILDDIR).

By default, VCCL compiles for all supported architectures. To speed up builds and reduce binary size, redefine NVCC_GENCODE (in makefiles/common.mk) to include only your target architecture(s):

```sh
# Example: build only for Hopper architecture (H100/H200, sm_90)
$ make -j80 src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## 📦 Packaging & Installation
To install VCCL on your system, build a package and install it as root:

```sh
# Debian/Ubuntu
sudo apt install -y build-essential devscripts debhelper fakeroot
make pkg.debian.build
ls build/pkg/deb/

# RedHat/CentOS
sudo yum install -y rpm-build rpmdevtools
make pkg.redhat.build
ls build/pkg/rpm/

# OS-agnostic tarball
make pkg.txz.build
ls build/pkg/txz/
```

## 🍽️ Testing
Tests for VCCL are maintained separately at NVIDIA NCCL Tests.

```sh
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Environment Varaibles on how to enable different features of VCCL

This part provides detailed reference for all VCCL environment variables, including functionality descriptions, default values, valid ranges, and usage examples.

### NCCL Compatibility

VCCL is fully compatible with NCCL 2.26.6-1 and supports all standard NCCL environment variables. For a complete list of NCCL environment variables, please refer to the [official NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

### VCCL-Specific Environment Variables

VCCL extends NCCL with the following additional environment variables:

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `NCCL_ENABLE_FAULT_TOLERANCE` | `0` | `0, 1` | Enable/disable fault tolerance functionality |
| `NCCL_PASS_SM` | `0` | `0, 1` | Enable SM Free mode for true communication-computation overlap |
| `NCCL_PSM_FORCE_ZEROCOPY` | `0` | `0, 1` | Force zero-copy mode (requires NCCL_PASS_SM=1) |
| `NCCL_PSM_BUFFER_SIZE` | auto | 1MB-1GB | Buffer size for SM-free P2P communication |
| `NCCL_PSM_P2P_NCHANNELS` | auto | 1-16 | Number of P2P channels for SM-free communication |
| `NCCL_TELEMETRY_ENABLE` | `0` | `0, 1` | Enable flow telemetry for real-time communication tracing |
| `TELEMETRY_WINDOWSIZE` | `50` | `10-1000` | Data window size for telemetry collection |
| `NCCL_TELEMETRY_LOG_PATH` | none | path string | Telemetry log output path |

### Usage Rules and Requirements

#### Fault Tolerance
- **Usage Limitation**: Requires NIC to be specified before use
- **Required NCCL Variables**: 
  - `NCCL_IB_HCA = mlx5_0:1,mlx5_1:1 ... mlx5_7:1` according to runtime environment
  - `NCCL_RETRY_COUNT` and `NCCL_TIMEOUT` use default values from NCCL 2.21: 7 and 18, respectively
- **Note**: These are standard NCCL environment variables, not VCCL-specific additions

#### SM Free Training
- **Requirement**: Must disable fault tolerance (`NCCL_ENABLE_FAULT_TOLERANCE=0`)
- **Limitation**: Fault tolerance is not yet supported with SM-free training
- **For SM-free training, the required changes to `mpi.sh` are shown below; for Megatron-side modifications, please refer to [sm-free-overlap-tarining](https://vccl-doc.readthedocs.io/en/latest/features/sm-free-overlap/).**
- **Required Settings**:
  - `NCCL_PASS_SM=1` (enable SM-free mode)
  - `NCCL_PSM_FORCE_ZEROCOPY=1` (force zero-copy)
  - `NCCL_PXN_DISABLE=1` (disable NCCL PXN)

#### Flow Telemetry
- **Requirement**: All three telemetry environment variables must be enabled together:
  - `NCCL_TELEMETRY_ENABLE=1`
  - `TELEMETRY_WINDOWSIZE` (set appropriate window size)
  - `NCCL_TELEMETRY_LOG_PATH` (specify output path)

#### Memory optimization
- **Required Settings**:
  - This feature is enabled by default

### Usage Examples

#### SM Free Training Configuration

```bash
# Optimized for SM-free training workloads
export NCCL_PASS_SM=1
export NCCL_PXN_DISABLE=1
export NCCL_PSM_FORCE_ZEROCOPY=1
export NCCL_IB_HCA=="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx6_3:1,mlx5_7:1"
export NCCL_ENABLE_FAULT_TOLERANCE=0
```

#### Fault-Tolerant Configuration
```bash
# Enable fault tolerance for production (without SM-free)
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx6_3:1,mlx5_7:1"
export NCCL_RETRY_COUNT=7
export NCCL_TIMEOUT=18
```

#### Flow Telemetry Configuration
```bash
# Enable flow telemetry for monitoring
export NCCL_TELEMETRY_ENABLE=1
export TELEMETRY_WINDOWSIZE=100
export NCCL_TELEMETRY_LOG_PATH=/tmp/vccl_telemetry
```

---

!!! tip "Configuration Tips"
    - VCCL supports all standard NCCL environment variables in addition to its own extensions
    - For optimal performance, use the default values unless you have specific requirements
    - Enable telemetry in development environments to monitor communication patterns
    - Use fault tolerance in production environments for reliability (but not with SM-free training)

!!! warning "Important Limitations"
    - **SM Free Training**: Cannot be used with fault tolerance (`NCCL_ENABLE_FAULT_TOLERANCE=0` required)
    - **Fault Tolerance**: Requires `NCCL_IB_HCA` to be set with your specific NIC configuration
    - **Flow Telemetry**: All three telemetry variables must be set together

!!! info "Performance Tuning"
    The optimal configuration depends on your specific hardware and workload. Start with the default values and adjust based on your performance requirements.

## 🙇‍♂️ License & Acknowledgements

* This project is developed based on nccl_2.26.6-1, and retains upstream copyright and license information in the relevant files.

* See the LICENSE file for detailed terms.

* Thanks to the open-source community (including but not limited to NCCL and nccl-tests) for their outstanding work.
