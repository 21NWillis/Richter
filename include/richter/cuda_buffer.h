#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <utility>

template <typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count = 0) : m_ptr(nullptr), m_count(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&m_ptr, count * sizeof(T));
            if (err != cudaSuccess) {
                fprintf(stderr, "CudaBuffer allocation failed: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE); 
            }
        }
    }

    ~CudaBuffer() {
        if (m_ptr) {
            cudaFree(m_ptr);
            m_ptr = nullptr;
        }
    }

    CudaBuffer(CudaBuffer&& other) noexcept : m_ptr(other.m_ptr), m_count(other.m_count) {
        other.m_ptr = nullptr;
        other.m_count = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (m_ptr) cudaFree(m_ptr);
            m_ptr = other.m_ptr;
            m_count = other.m_count;
            other.m_ptr = nullptr;
            other.m_count = 0;
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* data() const { return m_ptr; }
    size_t size() const { return m_count; }
    size_t bytes() const { return m_count * sizeof(T); }

    /// Try to allocate without exiting on failure. Returns true on success.
    static bool tryAlloc(CudaBuffer& buf, size_t count) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            cudaGetLastError();  // clear the error
            return false;
        }
        // Free any existing allocation
        if (buf.m_ptr) cudaFree(buf.m_ptr);
        buf.m_ptr = ptr;
        buf.m_count = count;
        return true;
    }

    void zero() {
        if (m_ptr && m_count > 0) {
            cudaMemset(m_ptr, 0, bytes());
        }
    }

    void copyFromHost(const T* host_data, size_t count) {
        if (count > m_count) {
             fprintf(stderr, "CudaBuffer overflow in copyFromHost\n");
             exit(EXIT_FAILURE);
        }
        cudaMemcpy(m_ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyToHost(T* host_data, size_t count) const {
        if (count > m_count) {
             fprintf(stderr, "CudaBuffer overflow in copyToHost\n");
             exit(EXIT_FAILURE);
        }
        cudaMemcpy(host_data, m_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    void swap(CudaBuffer& other) noexcept {
        std::swap(m_ptr, other.m_ptr);
        std::swap(m_count, other.m_count);
    }

private:
    T* m_ptr;
    size_t m_count;
};
