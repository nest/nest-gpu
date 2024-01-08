#include <iostream>
#include "connect.h"
#include "conn12b.h"

__global__ void setMaxNodeNBitsKernel(int max_node_nbits,
				      int max_port_syn_nbits,
				      int max_delay_nbits,
				      int max_port_nbits,
				      uint port_syn_mask,
				      uint delay_mask,
				      uint source_mask,
				      uint target_mask,
				      uint port_mask)
{
  MaxNodeNBits = max_node_nbits;
  MaxPortSynNBits = max_port_syn_nbits;
  MaxDelayNBits = max_delay_nbits;
  MaxPortNBits = max_port_nbits;
  PortSynMask = port_syn_mask;
  DelayMask = delay_mask;
  SourceMask = source_mask;
  TargetMask = target_mask;
  PortMask = port_mask;
}

__global__ void setMaxSynNBitsKernel(int max_syn_nbits,
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
int ConnectionTemplate<conn12b_key, conn12b_struct>::setMaxNodeNBits
(int max_node_nbits)
{
  // maximum number of bits used to represent node index
  max_node_nbits_ = max_node_nbits;
  
  // maximum number of bits used to represent receptor port index
  // and synapse group index
  max_port_syn_nbits_ = 32 - max_node_nbits_;
  
  // maximum number of bits used to represent delays
  max_delay_nbits_ = max_port_syn_nbits_;

  // maximum number of bits used to represent receptor port index
  max_port_nbits_ = max_port_syn_nbits_ - max_syn_nbits_ - 1;

  // bit mask used to extract port and synapse group index  
  port_syn_mask_ = (1 << max_port_syn_nbits_) - 1;

  // bit mask used to extract delay
  delay_mask_ = port_syn_mask_;

  // bit mask used to extract source node index
  source_mask_ = ~delay_mask_;

  // bit mask used to extract target node index
  target_mask_ = source_mask_;

  // bit mask used to extract port index  
  port_mask_ = ((1 << max_port_nbits_) - 1) << (max_syn_nbits_ + 1);

  // call CUDA kernel to initialize variables in device memory
  setMaxNodeNBitsKernel<<<1,1>>>
    (max_node_nbits_, max_port_syn_nbits_, max_delay_nbits_, max_port_nbits_,
     port_syn_mask_, delay_mask_, source_mask_, target_mask_, port_mask_);
  
  DBGCUDASYNC;

  return 0;
}  

// Set maximum number of bits used to represent delay
// and other dependent variables
template<>
int ConnectionTemplate<conn12b_key, conn12b_struct>::setMaxDelayNBits
(int max_delay_nbits)
{
  return setMaxNodeNBits(32 - max_delay_nbits);
}

// Set maximum number of bits used to represent synapse group index
// and other dependent variables
template<>
int ConnectionTemplate<conn12b_key, conn12b_struct>::setMaxSynNBits
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
  setMaxSynNBitsKernel<<<1,1>>>(max_syn_nbits_, max_port_nbits_,
				syn_mask_, port_mask_);
  DBGCUDASYNC;

  return 0;
}  

template <>
void ConnectionTemplate<conn12b_key, conn12b_struct>::setConnSource
(conn12b_key &conn_key, inode_t source) {
  conn_key = (conn_key & (~source_mask_)) | (source << max_delay_nbits_);
}

template <>
int ConnectionTemplate<conn12b_key, conn12b_struct>::getConnDelay
(const conn12b_key &conn_key) {
  return conn_key & delay_mask_;
}

template <>
ConnectionTemplate<conn12b_key, conn12b_struct>::ConnectionTemplate()
{
  //std::cout << "In Connectiontemplate<conn12b_key, conn12b_struct> "
  //"specialized constructor\n";
  init();
  setMaxNodeNBits(20); // maximum number of nodes is 2^20
  //std::cout << "max_node_nbits_: " << max_node_nbits_ << "\n";
  setMaxSynNBits(6); // maximum number of synapse groups is 2^6
}

