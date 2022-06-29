#ifndef NEW_CONNECT_H
#define NEW_CONNECT_H

#include <curand.h>
#include <vector>

struct value_struct
{
  int target;
  float weight;
};

extern uint *d_ConnGroupIdx0;
extern __device__ uint *ConnGroupIdx0;

extern uint *d_ConnGroupNum;
extern __device__ uint *ConnGroupNum;

extern int64_t *d_ConnGroupIConn0;
extern __device__ int64_t *ConnGroupIConn0;

extern int64_t *d_ConnGroupNConn;
extern __device__ int64_t *ConnGroupNConn;

extern uint *d_ConnGroupDelay;
extern __device__ uint *ConnGroupDelay;

extern int64_t NConn;

extern const int64_t h_ConnBlockSize;
extern __device__ int64_t ConnBlockSize;

extern std::vector<uint*> KeySubarray;
extern std::vector<value_struct*> ValueSubarray;

extern __device__ value_struct** ConnectionArray;

int connect_fixed_total_number(curandGenerator_t &gen,
			       void *d_storage, float time_resolution,
			       std::vector<uint*> &key_subarray,
			       std::vector<value_struct*> &value_subarray,
			       int64_t &n_conn, int block_size,
			       int64_t total_num, int i_source0, int n_source,
			       int i_target0, int n_target, int port,
			       float weight_mean, float weight_std,
			       float delay_mean, float delay_std);

int connect_all_to_all(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<uint*> &key_subarray,
		       std::vector<value_struct*> &value_subarray,
		       int64_t &n_conn, int block_size,
		       int i_source0, int n_source,
		       int i_target0, int n_target, int port,
		       float weight_mean, float weight_std,
		       float delay_mean, float delay_std);


int organizeConnections(float time_resolution, uint n_node, int64_t n_conn,
			int64_t block_size,
			std::vector<uint*> &key_subarray,
			std::vector<value_struct*> &value_subarray);

int NewConnectInit();

#endif
