#include <iostream>
#include "connect.h"
#include "conn16b.h"

__global__ void setMaxDelayNBits16bKernel(int max_delay_nbits,
					  int max_port_syn_nbits,
					  int max_port_nbits,
					  uint port_syn_mask,
					  uint delay_mask,
					  uint source_mask,
					  uint port_mask)
{
  MaxDelayNBits = max_delay_nbits;
  MaxPortSynNBits = max_port_syn_nbits;
  MaxPortNBits = max_port_nbits;
  PortSynMask = port_syn_mask;
  DelayMask = delay_mask;
  SourceMask = source_mask;
  PortMask = port_mask;
}

__global__ void setMaxSynNBits16bKernel(int max_syn_nbits,
				     int max_port_nbits,
				     uint syn_mask,
				     uint port_mask)
{
  MaxSynNBits = max_syn_nbits;
  MaxPortNBits = max_port_nbits;
  SynMask = syn_mask;
  PortMask = port_mask;
}

// Set maximum number of bits used to represent node index
// and other dependent variables
template<>
int ConnectionTemplate<conn16b_key, conn16b_struct>::setMaxNodeNBits
(int max_node_nbits)
{
  std::cout << "Warning: number of bits representing node index is fixed "
    "to 32 and cannot be modified with conn16b connection type";
  
  return 0;
}

// Set maximum number of bits used to represent delay
// and other dependent variables
template<>
int ConnectionTemplate<conn16b_key, conn16b_struct>::setMaxDelayNBits
(int max_delay_nbits)
{
  // maximum number of bits used to represent node index
  max_delay_nbits_ = max_delay_nbits;
  
  // maximum number of bits used to represent receptor port index
  // and synapse group index
  max_port_syn_nbits_ = 32 - max_delay_nbits_;
  
  // maximum number of bits used to represent receptor port index
  max_port_nbits_ = max_port_syn_nbits_ - max_syn_nbits_ - 1;

  // bit mask used to extract port and synapse group index  
  port_syn_mask_ = (1 << max_port_syn_nbits_) - 1;

  // bit mask used to extract delay
  delay_mask_ = ((1 << max_delay_nbits_) - 1) << max_port_syn_nbits_;

  // bit mask used to extract source node index
  source_mask_ = 0xFFFFFFFF; 
  
  // bit mask used to extract port index  
  port_mask_ = ((1 << max_port_nbits_) - 1) << (max_syn_nbits_ + 1);

  // call CUDA kernel to initialize variables in device memory
  setMaxDelayNBits16bKernel<<<1,1>>>
    (max_delay_nbits_, max_port_syn_nbits_, max_port_nbits_,
     port_syn_mask_, delay_mask_, source_mask_, port_mask_);
  
  DBGCUDASYNC;

  return 0;
}  

// Set maximum number of bits used to represent synapse group index
// and other dependent variables
template<>
int ConnectionTemplate<conn16b_key, conn16b_struct>::setMaxSynNBits
(int max_syn_nbits)
{
  // maximum number of bits used to represent synapse group index
  max_syn_nbits_ = max_syn_nbits;

  // maximum number of bits used to represent receptor port index  
  max_port_nbits_ = max_port_syn_nbits_ - max_syn_nbits_ - 1;

  // bit mask used to extract synapse group index
  syn_mask_ = (1 << max_syn_nbits_) - 1;

  // bit mask used to extract port index  
  port_mask_ = ((1 << max_port_nbits_) - 1) << (max_syn_nbits_ + 1);

  // call CUDA kernel to initialize variables in device memory
  setMaxSynNBits16bKernel<<<1,1>>>(max_syn_nbits_, max_port_nbits_,
				   syn_mask_, port_mask_);
  DBGCUDASYNC;

  return 0;
}  

template <>
void ConnectionTemplate<conn16b_key, conn16b_struct>::setConnSource
(conn16b_key &conn_key, inode_t source) {
    conn_key = (conn_key & 0xFFFFFFFF) | ((uint64_t)source << 32);
}

template <>
int ConnectionTemplate<conn16b_key, conn16b_struct>::getConnDelay 
(const conn16b_key &conn_key) {
  return (int)((conn_key & delay_mask_) >> max_port_syn_nbits_);
}

template <>
ConnectionTemplate<conn16b_key, conn16b_struct>::ConnectionTemplate()
{
  //std::cout << "In Connectiontemplate<conn16b_key, conn16b_struct> "
  //"specialized constructor\n";
  init();
  max_node_nbits_ = 31;
  setMaxDelayNBits(14); // maximum number of bits for delays
  //std::cout << "max_node_nbits_: " << max_node_nbits_ << "\n";
  setMaxSynNBits(10); // maximum number of synapse groups is 2^10
}

