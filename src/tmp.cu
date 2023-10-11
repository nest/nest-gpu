

int poiss_gen::OrganizeConnections()
{
  typedef uint key_t;
  typedef regular_block_array<key_t> array_t;
  
  uint k = KeySubarray.size();
  int64_t n = NConn;
  int64_t block_size = h_ConnBlockSize;
  
  key_t **key_subarray = KeySubarray.data();
  
  array_t h_key_array;
  array_t d_key_array;

  h_key_array.data_pt = key_subarray;
  h_key_array.block_size = block_size;
  h_key_array.offset = 0;
  h_key_array.size = n;

  key_t **d_key_array_data_pt = NULL;
  CUDAMALLOCCTRL("&d_key_array_data_pt",&d_key_array_data_pt, k*sizeof(key_t*));
  gpuErrchk(cudaMemcpy(d_key_array_data_pt, key_subarray,
		       k*sizeof(key_t*), cudaMemcpyHostToDevice));

  d_key_array.data_pt = d_key_array_data_pt; //key_subarray;
  d_key_array.block_size = block_size;
  d_key_array.offset = 0;
  d_key_array.size = n;

  array_t h_subarray[k];
  for (uint i=0; i<k; i++) {
    h_subarray[i].h_data_pt = key_subarray;
    h_subarray[i].data_pt = d_key_array_data_pt; //key_subarray;
    h_subarray[i].block_size = block_size;
    h_subarray[i].offset = i * block_size;
    h_subarray[i].size = i<k-1 ? block_size : n-(k-1)*block_size;
  }

  array_t *d_subarray;
  CUDAMALLOCCTRL("&d_subarray",&d_subarray, k, sizeof(array_t));
  gpuErrchk(cudaMemcpyAsync(d_subarray, h_subarray,
			    k*sizeof(array_t), cudaMemcpyHostToDevice));

  
  
  int64_t h_num[k];
  int64_t *d_num;
  CUDAMALLOCCTRL("&d_num",&d_num, 2*k*sizeof(int64_t));
  int64_t *d_sum;
  CUDAMALLOCCTRL("&d_sum",&d_sum, 2*sizeof(int64_t));
  
  key_t h_thresh[2];
  key_t *d_thresh;
  CUDAMALLOCCTRL("&d_thresh",&d_thresh, 2*sizeof(key_t));
  
  int64_t *d_num0 = &d_num[0];
  int64_t *d_num1 = &d_num[k];
  int64_t *h_num0 = &h_num[0];
  int64_t *h_num1 = &h_num[k];
  

  h_thresh[0] = i_node_0_ << MaxPortNBits;
  h_thresh[1] = (i_node_0_ + n_node_) << MaxPortNBits;
  
  search_multi_down<key_t, array_t, 1024>
    (d_subarray, k, &d_thresh[0], d_num0, d_sum[0]);
  CUDASYNC
    
  search_multi_down<key_t, array_t, 1024>
    (d_subarray, k, &d_thresh[1], d_num1, d_sum[1]);
  CUDASYNC

  gpuErrchk(cudaMemcpy(h_num, d_num, 2*k*sizeof(int64_t),
		       cudaMemcpyDeviceToHost));
  int64_t n_conn;
  int64_t i_conn0 = 0;
  int64_t i_conn1 = 0;
  uint ib0 = 0;
  uint ib1 = 0;
  uint nb;
  for (uint i=0; i<k; i++) {
    if (h_num0[i] < block_size) {
      i_conn0 = block_size*i + h_num0[i];
      ib0 = i;
      break;
    }
  }
  for (uint i=0; i<k; i++) {
    if (h_num1[i] < block_size) {
      i_conn1 = block_size*i + h_num1[i];
      ib1 = i;
      break;
    }
  }
  n_conn = i_conn0 - i_conn1;
  if (n_conn>0) {
    key_t *d_poiss_key_array;
    CUDAMALLOCCTRL("&d_poiss_key_array",&d_poiss_key_array, n_conn*sizeof(key_t));
    
    int64_t offset = 0;
    for (uint ib=ib0; ib<=ib1; ib++) {
      if (ib==ib0 && ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array, key_subarray[ib] + h_num0[ib],
			     n_conn*sizeof(key_t), cudaMemcpyDeviceToDevice));
	break;
      }
      else if (ib==ib0) {
	offset = block_size - h_num0[ib];
	gpuErrchk(cudaMemcpy(d_poiss_key_array, key_subarray[ib] + h_num0[ib],
			     offset*sizeof(key_t),
			     cudaMemcpyDeviceToDevice));
      }
      else if (ib==ib1) {
	gpuErrchk(cudaMemcpy(d_poiss_key_array + offset,
			     key_subarray[ib] + h_num0[ib],
			     h_num1[i]*sizeof(key_t),
			     cudaMemcpyDeviceToDevice));
      }
      else {
	gpuErrchk(cudaMemcpy(d_poiss_key_array + offset,
			     key_subarray[ib],
			     block_size*sizeof(key_t),
			     cudaMemcpyDeviceToDevice));
      }
    }
    key_t *h_poiss_key_array = new key_t[n_conn];
    gpuErrchk(cudaMemcpy(h_poiss_key_array, d_poiss_key_array,
			 n_conn*sizeof(key_t),
			 cudaMemcpyDeviceToHost));
    printf("i_conn0: %ld\ti_conn1: %ld\tn_conn: %ld\n", i_conn0, i_conn1,
	   n_conn);
    int i_min = h_poiss_key_array[0] >> MaxPortNBits;
    int d_min = h_poiss_key_array[0] & PortMask;
    int i_max = h_poiss_key_array[n_conn - 1] >> MaxPortNBits;
    int d_max = h_poiss_key_array[n_conn - 1] & PortMask;
    printf("i_min: %d\ti_max: %d\td_min: %d\td_max: %d\n"
	   i_min, i_max, d_min, d_max);
  }
  
  gpuErrchk(cudaFree(d_key_array_data_pt));
  gpuErrchk(cudaFree(d_subarray));
  gpuErrchk(cudaFree(d_num));
  gpuErrchk(cudaFree(d_sum));
  gpuErrchk(cudaFree(d_thresh));

  return 0;
}

__global__ void SendDirectSpikes(int64_t n_conn, int64_t i_conn_0,
				 int64_t block_size, int n_node,
				 float *rate_arr, int max_delay_num,
				 float time_resolution)
{
  i_conn_rel = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_conn_rel >= n_conn) {
    return 0;
  }
  uint source_delay = PoissKeyArray[i_conn_rel];
  int i_source = source_delay >> MaxPortNBits;
  int i_delay = source_delay & PortMask;
  int id = (NESTGPUTimeIdx - i_delay + 1) % max_delay_num;
  float r = rate_arr[id*n_node + i_source];
  float height = r*time_resolution;
  
  int64_t i_conn = i_conn_0 + i_conn_rel;
  int i_block = (int)(i_conn / block_size);
  int64_t i_block_conn = i_conn % block_size;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  uint target_port = conn.target_port;
  int i_target = target_port >> MaxPortNBits;
  uint port = target_port & PortMask;
  float weight = conn.weight;

  int i_group=NodeGroupMap[i_target];
  int i = port*NodeGroupArray[i_group].n_node_ + i_target
    - NodeGroupArray[i_group].i_node_0_;
  double d_val = (double)(height*weight);
  atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);

}
