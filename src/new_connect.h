#ifndef NEW_CONNECT_H
#define NEW_CONNECT_H

#include <curand.h>
#include <vector>

struct connection_struct
{
  int target_port;
  float weight;
};

extern uint h_MaxNodeNBits;
extern __device__ uint MaxNodeNBits;

extern uint h_MaxPortNBits;
extern __device__ uint MaxPortNBits;

extern uint h_PortMask;
extern __device__ uint PortMask;

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

extern int64_t h_ConnBlockSize;
extern __device__ int64_t ConnBlockSize;

extern uint h_MaxDelayNum;

extern std::vector<uint*> KeySubarray;
extern std::vector<connection_struct*> ConnectionSubarray;

extern __device__ connection_struct** ConnectionArray;

int setMaxNodeNBits(int max_node_nbits);

int connect_fixed_total_number(curandGenerator_t &gen,
			       void *d_storage, float time_resolution,
			       std::vector<uint*> &key_subarray,
			       std::vector<connection_struct*> &conn_subarray,
			       int64_t &n_conn, int block_size,
			       int64_t total_num, int i_source0, int n_source,
			       int i_target0, int n_target, int port,
			       float weight_mean, float weight_std,
			       float delay_mean, float delay_std);

int connect_all_to_all(curandGenerator_t &gen,
		       void *d_storage, float time_resolution,
		       std::vector<uint*> &key_subarray,
		       std::vector<connection_struct*> &conn_subarray,
		       int64_t &n_conn, int block_size,
		       int i_source0, int n_source,
		       int i_target0, int n_target, int port,
		       float weight_mean, float weight_std,
		       float delay_mean, float delay_std);


int organizeConnections(float time_resolution, uint n_node, int64_t n_conn,
			int64_t block_size,
			std::vector<uint*> &key_subarray,
			std::vector<connection_struct*> &conn_subarray);

int NewConnectInit();

#endif
