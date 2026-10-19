__host__ void init_lm_maps(int p) {
    int l_map[MAX_TERMS];
    int m_map[MAX_TERMS];
    for (int l = 0; l <= p; l++) {
        for (int m = 0; m <= l; m++) {
            int idx = l * (l + 1) / 2 + m;
            l_map[idx] = l;
            m_map[idx] = m;
        }
    }
    
    cudaMemcpyToSymbol(d_l_map, l_map, MAX_TERMS * sizeof(int));
    cudaMemcpyToSymbol(d_m_map, m_map, MAX_TERMS * sizeof(int));
}

__global__ void m2mKernel(
    int total_nodes, int p,
    const int* __restrict__ d_child_idx,
    const double* __restrict__ d_center_x,
    const double* __restrict__ d_center_y,
    const double* __restrict__ d_center_z,
    double* __restrict__ d_M_real,
    double* __restrict__ d_M_imag,
    int level_start
) {
    int parent_idx = level_start + blockIdx.x;

    if (parent_idx >= total_nodes) return;

    int first_child = d_child_idx[parent_idx];

    if (first_child == -1) return;

    int tid = threadIdx.x;
    int num_terms = (p + 1) * (p + 2) / 2;

    // shared memory arrays to store child's multipole
    __shared__ double s_child_M_real[MAX_TERMS * 8];
    __shared__ double s_child_M_imag[MAX_TERMS * 8];

    int total_child_terms = num_terms * 8;

    for (int i = tid; i < total_child_terms; i += blockDim.x) {
        int c_idx = i / num_terms;
        int term_idx = i % num_terms;

        int global_child_idx = first_child + c_idx;
        int global_offset = global_child_idx * num_terms + term_idx;

        s_child_M_real[i] = d_M_real[global_offset];
        s_child_M_imag[i] = d_M_imag[global_offset];
    }

    __syncthreads();

    for (int term_idx = tid; term_idx < num_terms; term_idx += blockDim.x) {
        int l = d_l_map[term_idx];
        int m = d_m_map[term_idx];

        double my_parent_term_real = 0.0;
        double my_parent_term_imag = 0.0;

        double cx = d_center_x[parent_idx];
        double cy = d_center_y[parent_idx];
        double cz = d_center_z[parent_idx];

        for (int i = 0; i < 8; i++) {
            int c_idx = first_child + i;

            double dx = d_center_x[c_idx] - cx;
            double dy = d_center_y[c_idx] - cy;
            double dz = d_center_z[c_idx] - cz;
            double r2 = dx*dx + dy*dy + dz*dz;

            double child_R_real[MAX_TERMS];
            double child_R_imag[MAX_TERMS];

            child_R_real[0] = 1.0;
            child_R_imag[0] = 0.0;

            for (int current_l = 1; current_l <= l; current_l++) {
                int idx_diag = current_l * (current_l + 1) / 2 + current_l;
                int idx_prev = current_l * (current_l - 1) / 2 + (current_l - 1);

                double factor = -1.0 / (2.0 * current_l);

                double Rp = child_R_real[idx_prev];
                double Ip = child_R_imag[idx_prev];

                child_R_real[idx_diag] = factor * (dx * Rp - dy * Ip);
                child_R_imag[idx_diag] = factor * (dy * Ip + dx * Rp);

                int idx_off_diag = current_l * (current_l + 1) / 2 + (current_l - 1);

                child_R_real[idx_off_diag] = dz * Rp;
                child_R_imag[idx_off_diag] = dz * Ip;

                if (current_l >= 2) {
                    for (int current_m = current_l - 2; current_m >= 0; current_m--) {
                        int idx_lm = current_l * (current_l + 1) / 2 + current_m;
                        int prev1 = current_l * (current_l - 1) / 2 + current_m;
                        int prev2 = (current_l - 2) * (current_l - 1) / 2 + current_m;

                        double denom = static_cast<double>((current_l + current_m) * (current_l - current_m));
                        double inv_denom = 1.0 / denom;

                        child_R_real[idx_lm] = ((2.0 * current_l - 1) * dz * child_R_real[prev1] - r2 * child_R_real[prev2]) * inv_denom;
                        child_R_imag[idx_lm] = ((2.0 * current_l - 1) * dz * child_R_imag[prev1] - r2 * child_R_imag[prev2]) * inv_denom;
                    }
                }
            }

            for (int j = 0; j <= l; j++) {
                for (int k = 0; k <= j; k++) {
                    int delta_m = m - k;
                    int abs_delta_m = abs(delta_m);
                    int child_term_idx = (l - j) * (l - j + 1) / 2 + abs_delta_m;
                    int shared_child_offset = i * num_terms + child_term_idx;

                    double C_M_real = s_child_M_real[shared_child_offset];
                    double C_M_imag = s_child_M_imag[shared_child_offset];

                    if (delta_m < 0) {
                        C_M_imag = -C_M_imag; // Conjugate

                        if (abs_delta_m % 2 != 0) { // (-1)^m {odd becomes negative}
                            C_M_real = -C_M_real;
                            C_M_imag = -C_M_imag;
                        }
                    }

                    int shift_term_idx = j * (j + 1) / 2 + k;
                    double R_real = child_R_real[shift_term_idx];
                    double R_imag = child_R_imag[shift_term_idx];

                    my_parent_term_real += (C_M_real * R_real - C_M_imag * R_imag);
                    my_parent_term_imag += (C_M_real * R_imag + C_M_imag * R_real);
                }
            }
        }

        // Write the fully accumulated term to global memory
        int global_parent_offset = parent_idx * num_terms + term_idx;
        d_M_real[global_parent_offset] = my_parent_term_real;
        d_M_imag[global_parent_offset] = my_parent_term_imag;
    }
}

__host__ void computeM2MCUDA(
    int total_nodes, int p,
    const int* d_child_idx, const double* d_center_x,
    const double* d_center_y, const double* d_center_z,
    double* d_M_real, double* d_M_imag,
    const std::vector<int>& level_bounds
) {
    int max_level = level_bounds.size() - 2;

    for (int level = max_level; level >= 0; level--) {
        int level_start = level_bounds[level];
        int level_end = level_bounds[level + 1];
        int num_nodes_in_level = level_end - level_start;

        if (num_nodes_in_level == 0) continue;

        dim3 block(128);
        dim3 grid(num_nodes_in_level);

        m2mKernel<<<grid, block>>>(
            total_nodes, p,
            d_child_idx, d_center_x,
            d_center_y, d_center_z,
            d_M_real, d_M_imag,
            level_start
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}