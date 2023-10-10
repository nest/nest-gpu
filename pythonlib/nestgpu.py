""" Python interface for NESTGPU"""
import sys, platform
import ctypes, ctypes.util
import os
import unicodedata
import gc


print('\n              -- NEST GPU --\n')
print('  Copyright (C) 2021 The NEST Initiative\n')
print(' This program is provided AS IS and comes with')
print(' NO WARRANTY. See the file LICENSE for details.\n')
print(' Homepage: https://github.com/nest/nest-gpu')
print()


lib_path=os.environ["NESTGPU_LIB"]
_nestgpu=ctypes.CDLL(lib_path)

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_char_p = ctypes.POINTER(ctypes.c_char)
c_void_p = ctypes.c_void_p
c_int_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
c_float_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
c_float_ppp = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

class NodeSeq(object):
    def __init__(self, i0, n=1):
        if i0 == None:
            i0 = 0
            n = -1
        self.i0 = i0
        self.n = n

    def Subseq(self, first, last):
        if last<0 and last>=-self.n:
            last = last%self.n
        if first<0 | last<first:
            raise ValueError("Sequence subset range error")
        if last>=self.n:
            raise ValueError("Sequence subset out of range")
        return NodeSeq(self.i0 + first, last - first + 1)
    def __getitem__(self, i):
        if type(i)==slice:
            if i.step != None:
                raise ValueError("Subsequence cannot have a step")
            return self.Subseq(i.start, i.stop-1)
 
        if i<-self.n:
            raise ValueError("Sequence index error")
        if i>=self.n:
            raise ValueError("Sequence index out of range")
        if i<0:
            i = i%self.n
        return self.i0 + i
    def ToList(self):
        return list(range(self.i0, self.i0 + self.n))
    def __len__(self):
        return self.n

class RemoteNodeSeq(object):
    def __init__(self, i_host=0, node_seq=NodeSeq(None)):
        self.i_host = i_host
        self.node_seq = node_seq

class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

class SynGroup(object):
    def __init__(self, i_syn_group):
        self.i_syn_group = i_syn_group

def to_byte_str(s):
    if type(s)==str:
        return s.encode('ascii')
    elif type(s)==bytes:
        return s
    else:
        raise ValueError("Variable cannot be converted to string")

def to_def_str(s):
    if (sys.version_info >= (3, 0)):
        return s.decode("utf-8")
    else:
        return s

def waitenter(val):
    if (sys.version_info >= (3, 0)):
        return input(val)
    else:
        return raw_input(val)
    
conn_rule_name = ("one_to_one", "all_to_all", "fixed_total_number",
                  "fixed_indegree", "fixed_outdegree")
    
NESTGPU_GetErrorMessage = _nestgpu.NESTGPU_GetErrorMessage
NESTGPU_GetErrorMessage.restype = ctypes.POINTER(ctypes.c_char)
def GetErrorMessage():
    "Get error message from NESTGPU exception"
    message = ctypes.cast(NESTGPU_GetErrorMessage(), ctypes.c_char_p).value
    return message
 
NESTGPU_GetErrorCode = _nestgpu.NESTGPU_GetErrorCode
NESTGPU_GetErrorCode.restype = ctypes.c_ubyte
def GetErrorCode():
    "Get error code from NESTGPU exception"
    return NESTGPU_GetErrorCode()
 
NESTGPU_SetOnException = _nestgpu.NESTGPU_SetOnException
NESTGPU_SetOnException.argtypes = (ctypes.c_int,)
def SetOnException(on_exception):
    "Define whether handle exceptions (1) or exit (0) in case of errors"
    return NESTGPU_SetOnException(ctypes.c_int(on_exception))

SetOnException(1)

NESTGPU_SetRandomSeed = _nestgpu.NESTGPU_SetRandomSeed
NESTGPU_SetRandomSeed.argtypes = (ctypes.c_ulonglong,)
NESTGPU_SetRandomSeed.restype = ctypes.c_int
def SetRandomSeed(seed):
    "Set seed for random number generation"
    ret = NESTGPU_SetRandomSeed(ctypes.c_ulonglong(seed))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetTimeResolution = _nestgpu.NESTGPU_SetTimeResolution
NESTGPU_SetTimeResolution.argtypes = (ctypes.c_float,)
NESTGPU_SetTimeResolution.restype = ctypes.c_int
def SetTimeResolution(time_res):
    "Set time resolution in ms"
    ret = NESTGPU_SetTimeResolution(ctypes.c_float(time_res))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetTimeResolution = _nestgpu.NESTGPU_GetTimeResolution
NESTGPU_GetTimeResolution.restype = ctypes.c_float
def GetTimeResolution():
    "Get time resolution in ms"
    ret = NESTGPU_GetTimeResolution()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetMaxSpikeBufferSize = _nestgpu.NESTGPU_SetMaxSpikeBufferSize
NESTGPU_SetMaxSpikeBufferSize.argtypes = (ctypes.c_int,)
NESTGPU_SetMaxSpikeBufferSize.restype = ctypes.c_int
def SetMaxSpikeBufferSize(max_size):
    "Set maximum size of spike buffer per node"
    ret = NESTGPU_SetMaxSpikeBufferSize(ctypes.c_int(max_size))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetMaxSpikeBufferSize = _nestgpu.NESTGPU_GetMaxSpikeBufferSize
NESTGPU_GetMaxSpikeBufferSize.restype = ctypes.c_int
def GetMaxSpikeBufferSize():
    "Get maximum size of spike buffer per node"
    ret = NESTGPU_GetMaxSpikeBufferSize()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetSimTime = _nestgpu.NESTGPU_SetSimTime
NESTGPU_SetSimTime.argtypes = (ctypes.c_float,)
NESTGPU_SetSimTime.restype = ctypes.c_int
def SetSimTime(sim_time):
    "Set neural activity simulated time in ms"
    ret = NESTGPU_SetSimTime(ctypes.c_float(sim_time))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetVerbosityLevel = _nestgpu.NESTGPU_SetVerbosityLevel
NESTGPU_SetVerbosityLevel.argtypes = (ctypes.c_int,)
NESTGPU_SetVerbosityLevel.restype = ctypes.c_int
def SetVerbosityLevel(verbosity_level):
    "Set verbosity level"
    ret = NESTGPU_SetVerbosityLevel(ctypes.c_int(verbosity_level))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_Create = _nestgpu.NESTGPU_Create
NESTGPU_Create.argtypes = (c_char_p, ctypes.c_int, ctypes.c_int)
NESTGPU_Create.restype = ctypes.c_int
def Create(model_name, n_node=1, n_ports=1, status_dict=None):
    "Create a neuron group"
    if (type(status_dict)==dict):
        node_group = Create(model_name, n_node, n_ports)
        SetStatus(node_group, status_dict)
        return node_group
        
    elif status_dict!=None:
        raise ValueError("Wrong argument in Create")
    
    c_model_name = ctypes.create_string_buffer(to_byte_str(model_name), len(model_name)+1)
    i_node =NESTGPU_Create(c_model_name, ctypes.c_int(n_node), ctypes.c_int(n_ports))
    ret = NodeSeq(i_node, n_node)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_CreatePoissonGenerator = _nestgpu.NESTGPU_CreatePoissonGenerator
NESTGPU_CreatePoissonGenerator.argtypes = (ctypes.c_int, ctypes.c_float)
NESTGPU_CreatePoissonGenerator.restype = ctypes.c_int
def CreatePoissonGenerator(n_node, rate):
    "Create a poisson-distributed spike generator"
    i_node = NESTGPU_CreatePoissonGenerator(ctypes.c_int(n_node), ctypes.c_float(rate)) 
    ret = NodeSeq(i_node, n_node)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_CreateRecord = _nestgpu.NESTGPU_CreateRecord
NESTGPU_CreateRecord.argtypes = (c_char_p, ctypes.POINTER(c_char_p), c_int_p, c_int_p, ctypes.c_int)
NESTGPU_CreateRecord.restype = ctypes.c_int
def CreateRecord(file_name, var_name_list, i_node_list, i_port_list):
    "Create a record of neuron variables"
    n_node = len(i_node_list)
    c_file_name = ctypes.create_string_buffer(to_byte_str(file_name), len(file_name)+1)    
    array_int_type = ctypes.c_int * n_node
    array_char_pt_type = c_char_p * n_node
    c_var_name_list=[]
    for i in range(n_node):
        c_var_name = ctypes.create_string_buffer(to_byte_str(var_name_list[i]), len(var_name_list[i])+1)
        c_var_name_list.append(c_var_name)

    ret = NESTGPU_CreateRecord(c_file_name,
                                 array_char_pt_type(*c_var_name_list),
                                 array_int_type(*i_node_list),
                                 array_int_type(*i_port_list),
                                 ctypes.c_int(n_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetRecordDataRows = _nestgpu.NESTGPU_GetRecordDataRows
NESTGPU_GetRecordDataRows.argtypes = (ctypes.c_int,)
NESTGPU_GetRecordDataRows.restype = ctypes.c_int
def GetRecordDataRows(i_record):
    "Get record n. of rows"
    ret = NESTGPU_GetRecordDataRows(ctypes.c_int(i_record))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetRecordDataColumns = _nestgpu.NESTGPU_GetRecordDataColumns
NESTGPU_GetRecordDataColumns.argtypes = (ctypes.c_int,)
NESTGPU_GetRecordDataColumns.restype = ctypes.c_int
def GetRecordDataColumns(i_record):
    "Get record n. of columns"
    ret = NESTGPU_GetRecordDataColumns(ctypes.c_int(i_record))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetRecordData = _nestgpu.NESTGPU_GetRecordData
NESTGPU_GetRecordData.argtypes = (ctypes.c_int,)
NESTGPU_GetRecordData.restype = ctypes.POINTER(c_float_p)
def GetRecordData(i_record):
    "Get record data"
    data_arr_pt = NESTGPU_GetRecordData(ctypes.c_int(i_record))
    nr = GetRecordDataRows(i_record)
    nc = GetRecordDataColumns(i_record)
    data_list = []
    for ir in range(nr):
        row_list = []
        for ic in range(nc):
            row_list.append(data_arr_pt[ir][ic])
            
        data_list.append(row_list)
        
    ret = data_list    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronScalParam = _nestgpu.NESTGPU_SetNeuronScalParam
NESTGPU_SetNeuronScalParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NESTGPU_SetNeuronScalParam.restype = ctypes.c_int
def SetNeuronScalParam(i_node, n_node, param_name, val):
    "Set neuron scalar parameter value"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NESTGPU_SetNeuronScalParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronArrayParam = _nestgpu.NESTGPU_SetNeuronArrayParam
NESTGPU_SetNeuronArrayParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NESTGPU_SetNeuronArrayParam.restype = ctypes.c_int
def SetNeuronArrayParam(i_node, n_node, param_name, param_list):
    "Set neuron array parameter value"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    array_size = len(param_list)
    array_float_type = ctypes.c_float * array_size
    ret = NESTGPU_SetNeuronArrayParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name,
                                       array_float_type(*param_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronPtScalParam = _nestgpu.NESTGPU_SetNeuronPtScalParam
NESTGPU_SetNeuronPtScalParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NESTGPU_SetNeuronPtScalParam.restype = ctypes.c_int
def SetNeuronPtScalParam(nodes, param_name, val):
    "Set neuron list scalar parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    ret = NESTGPU_SetNeuronPtScalParam(node_pt,
                                         ctypes.c_int(n_node), c_param_name,
                                         ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronPtArrayParam = _nestgpu.NESTGPU_SetNeuronPtArrayParam
NESTGPU_SetNeuronPtArrayParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NESTGPU_SetNeuronPtArrayParam.restype = ctypes.c_int
def SetNeuronPtArrayParam(nodes, param_name, param_list):
    "Set neuron list array parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    
    array_size = len(param_list)
    array_float_type = ctypes.c_float * array_size
    ret = NESTGPU_SetNeuronPtArrayParam(node_pt,
                                          ctypes.c_int(n_node),
                                          c_param_name,
                                          array_float_type(*param_list),
                                          ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_IsNeuronScalParam = _nestgpu.NESTGPU_IsNeuronScalParam
NESTGPU_IsNeuronScalParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronScalParam.restype = ctypes.c_int
def IsNeuronScalParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NESTGPU_IsNeuronScalParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_IsNeuronPortParam = _nestgpu.NESTGPU_IsNeuronPortParam
NESTGPU_IsNeuronPortParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronPortParam.restype = ctypes.c_int
def IsNeuronPortParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NESTGPU_IsNeuronPortParam(ctypes.c_int(i_node), c_param_name)!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_IsNeuronArrayParam = _nestgpu.NESTGPU_IsNeuronArrayParam
NESTGPU_IsNeuronArrayParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronArrayParam.restype = ctypes.c_int
def IsNeuronArrayParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NESTGPU_IsNeuronArrayParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_IsNeuronGroupParam = _nestgpu.NESTGPU_IsNeuronGroupParam
NESTGPU_IsNeuronGroupParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronGroupParam.restype = ctypes.c_int
def IsNeuronGroupParam(i_node, param_name):
    "Check name of neuron scalar parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NESTGPU_IsNeuronGroupParam(ctypes.c_int(i_node), c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronIntVar = _nestgpu.NESTGPU_SetNeuronIntVar
NESTGPU_SetNeuronIntVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_int)
NESTGPU_SetNeuronIntVar.restype = ctypes.c_int
def SetNeuronIntVar(i_node, n_node, var_name, val):
    "Set neuron integer variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = NESTGPU_SetNeuronIntVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_int(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronScalVar = _nestgpu.NESTGPU_SetNeuronScalVar
NESTGPU_SetNeuronScalVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                         c_char_p, ctypes.c_float)
NESTGPU_SetNeuronScalVar.restype = ctypes.c_int
def SetNeuronScalVar(i_node, n_node, var_name, val):
    "Set neuron scalar variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = NESTGPU_SetNeuronScalVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronArrayVar = _nestgpu.NESTGPU_SetNeuronArrayVar
NESTGPU_SetNeuronArrayVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, c_float_p, ctypes.c_int)
NESTGPU_SetNeuronArrayVar.restype = ctypes.c_int
def SetNeuronArrayVar(i_node, n_node, var_name, var_list):
    "Set neuron array variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    array_size = len(var_list)
    array_float_type = ctypes.c_float * array_size
    ret = NESTGPU_SetNeuronArrayVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name,
                                       array_float_type(*var_list),
                                       ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronPtIntVar = _nestgpu.NESTGPU_SetNeuronPtIntVar
NESTGPU_SetNeuronPtIntVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_int)
NESTGPU_SetNeuronPtIntVar.restype = ctypes.c_int
def SetNeuronPtIntVar(nodes, var_name, val):
    "Set neuron list integer variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    ret = NESTGPU_SetNeuronPtIntVar(node_pt,
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_int(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronPtScalVar = _nestgpu.NESTGPU_SetNeuronPtScalVar
NESTGPU_SetNeuronPtScalVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, ctypes.c_float)
NESTGPU_SetNeuronPtScalVar.restype = ctypes.c_int
def SetNeuronPtScalVar(nodes, var_name, val):
    "Set neuron list scalar variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    ret = NESTGPU_SetNeuronPtScalVar(node_pt,
                                       ctypes.c_int(n_node), c_var_name,
                                       ctypes.c_float(val)) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronPtArrayVar = _nestgpu.NESTGPU_SetNeuronPtArrayVar
NESTGPU_SetNeuronPtArrayVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p, c_float_p,
                                           ctypes.c_int)
NESTGPU_SetNeuronPtArrayVar.restype = ctypes.c_int
def SetNeuronPtArrayVar(nodes, var_name, var_list):
    "Set neuron list array variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                             len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)

    array_size = len(var_list)
    array_float_type = ctypes.c_float * array_size
    ret = NESTGPU_SetNeuronPtArrayVar(node_pt,
                                        ctypes.c_int(n_node),
                                        c_var_name,
                                        array_float_type(*var_list),
                                        ctypes.c_int(array_size))  
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_IsNeuronIntVar = _nestgpu.NESTGPU_IsNeuronIntVar
NESTGPU_IsNeuronIntVar.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronIntVar.restype = ctypes.c_int
def IsNeuronIntVar(i_node, var_name):
    "Check name of neuron integer variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    ret = (NESTGPU_IsNeuronIntVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_IsNeuronScalVar = _nestgpu.NESTGPU_IsNeuronScalVar
NESTGPU_IsNeuronScalVar.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronScalVar.restype = ctypes.c_int
def IsNeuronScalVar(i_node, var_name):
    "Check name of neuron scalar variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    ret = (NESTGPU_IsNeuronScalVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_IsNeuronPortVar = _nestgpu.NESTGPU_IsNeuronPortVar
NESTGPU_IsNeuronPortVar.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronPortVar.restype = ctypes.c_int
def IsNeuronPortVar(i_node, var_name):
    "Check name of neuron scalar variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = (NESTGPU_IsNeuronPortVar(ctypes.c_int(i_node), c_var_name)!= 0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_IsNeuronArrayVar = _nestgpu.NESTGPU_IsNeuronArrayVar
NESTGPU_IsNeuronArrayVar.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsNeuronArrayVar.restype = ctypes.c_int
def IsNeuronArrayVar(i_node, var_name):
    "Check name of neuron array variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = (NESTGPU_IsNeuronArrayVar(ctypes.c_int(i_node), c_var_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronParamSize = _nestgpu.NESTGPU_GetNeuronParamSize
NESTGPU_GetNeuronParamSize.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_GetNeuronParamSize.restype = ctypes.c_int
def GetNeuronParamSize(i_node, param_name):
    "Get neuron parameter array size"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NESTGPU_GetNeuronParamSize(ctypes.c_int(i_node), c_param_name) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronParam = _nestgpu.NESTGPU_GetNeuronParam
NESTGPU_GetNeuronParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                     c_char_p)
NESTGPU_GetNeuronParam.restype = c_float_p
def GetNeuronParam(i_node, n_node, param_name):
    "Get neuron parameter value"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    data_pt = NESTGPU_GetNeuronParam(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_param_name)

    array_size = GetNeuronParamSize(i_node, param_name)
    data_list = []
    for i_node in range(n_node):
        if (array_size>1):
            row_list = []
            for i in range(array_size):
                row_list.append(data_pt[i_node*array_size + i])
        else:
            row_list = data_pt[i_node]
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronPtParam = _nestgpu.NESTGPU_GetNeuronPtParam
NESTGPU_GetNeuronPtParam.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p)
NESTGPU_GetNeuronPtParam.restype = c_float_p
def GetNeuronPtParam(nodes, param_name):
    "Get neuron list scalar parameter value"
    n_node = len(nodes)
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NESTGPU_GetNeuronPtParam(node_pt,
                                         ctypes.c_int(n_node), c_param_name)
    array_size = GetNeuronParamSize(nodes[0], param_name)

    data_list = []
    for i_node in range(n_node):
        if (array_size>1):
            row_list = []
            for i in range(array_size):
                row_list.append(data_pt[i_node*array_size + i])
        else:
            row_list = data_pt[i_node]
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetArrayParam = _nestgpu.NESTGPU_GetArrayParam
NESTGPU_GetArrayParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_GetArrayParam.restype = c_float_p
def GetArrayParam(i_node, n_node, param_name):
    "Get neuron array parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    data_list = []
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        row_list = []
        data_pt = NESTGPU_GetArrayParam(ctypes.c_int(i_node1), c_param_name)
        array_size = GetNeuronParamSize(i_node1, param_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetNeuronListArrayParam(node_list, param_name):
    "Get neuron array parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    data_list = []
    for i_node in node_list:
        row_list = []
        data_pt = NESTGPU_GetArrayParam(ctypes.c_int(i_node), c_param_name)
        array_size = GetNeuronParamSize(i_node, param_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronGroupParam = _nestgpu.NESTGPU_GetNeuronGroupParam
NESTGPU_GetNeuronGroupParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_GetNeuronGroupParam.restype = ctypes.c_float
def GetNeuronGroupParam(i_node, param_name):
    "Check name of neuron group parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NESTGPU_GetNeuronGroupParam(ctypes.c_int(i_node), c_param_name)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret



#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NESTGPU_GetNeuronVarSize = _nestgpu.NESTGPU_GetNeuronVarSize
NESTGPU_GetNeuronVarSize.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_GetNeuronVarSize.restype = ctypes.c_int
def GetNeuronVarSize(i_node, var_name):
    "Get neuron variable array size"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name), len(var_name)+1)
    ret = NESTGPU_GetNeuronVarSize(ctypes.c_int(i_node), c_var_name)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronIntVar = _nestgpu.NESTGPU_GetNeuronIntVar
NESTGPU_GetNeuronIntVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                      c_char_p)
NESTGPU_GetNeuronIntVar.restype = c_int_p
def GetNeuronIntVar(i_node, n_node, var_name):
    "Get neuron integer variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_pt = NESTGPU_GetNeuronIntVar(ctypes.c_int(i_node),
                                        ctypes.c_int(n_node), c_var_name)

    data_list = []
    for i_node in range(n_node):
        data_list.append([data_pt[i_node]])
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronVar = _nestgpu.NESTGPU_GetNeuronVar
NESTGPU_GetNeuronVar.argtypes = (ctypes.c_int, ctypes.c_int,
                                     c_char_p)
NESTGPU_GetNeuronVar.restype = c_float_p
def GetNeuronVar(i_node, n_node, var_name):
    "Get neuron variable value"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_pt = NESTGPU_GetNeuronVar(ctypes.c_int(i_node),
                                       ctypes.c_int(n_node), c_var_name)

    array_size = GetNeuronVarSize(i_node, var_name)

    data_list = []
    for i_node in range(n_node):
        if (array_size>1):
            row_list = []
            for i in range(array_size):
                row_list.append(data_pt[i_node*array_size + i])
        else:
            row_list = data_pt[i_node]
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNeuronPtIntVar = _nestgpu.NESTGPU_GetNeuronPtIntVar
NESTGPU_GetNeuronPtIntVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                        c_char_p)
NESTGPU_GetNeuronPtIntVar.restype = c_int_p
def GetNeuronPtIntVar(nodes, var_name):
    "Get neuron list integer variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NESTGPU_GetNeuronPtIntVar(node_pt,
                                          ctypes.c_int(n_node), c_var_name)
    data_list = []
    for i_node in range(n_node):
        data_list.append([data_pt[i_node]])
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetNeuronPtVar = _nestgpu.NESTGPU_GetNeuronPtVar
NESTGPU_GetNeuronPtVar.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                           c_char_p)
NESTGPU_GetNeuronPtVar.restype = c_float_p
def GetNeuronPtVar(nodes, var_name):
    "Get neuron list scalar variable value"
    n_node = len(nodes)
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    node_arr = (ctypes.c_int * len(nodes))(*nodes)
    node_pt = ctypes.cast(node_arr, ctypes.c_void_p)
    data_pt = NESTGPU_GetNeuronPtVar(node_pt,
                                       ctypes.c_int(n_node), c_var_name)
    array_size = GetNeuronVarSize(nodes[0], var_name)

    data_list = []

    for i_node in range(n_node):
        if (array_size>1):
            row_list = []
            for i in range(array_size):
                row_list.append(data_pt[i_node*array_size + i])
        else:
            row_list = data_pt[i_node]
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetArrayVar = _nestgpu.NESTGPU_GetArrayVar
NESTGPU_GetArrayVar.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_GetArrayVar.restype = c_float_p
def GetArrayVar(i_node, n_node, var_name):
    "Get neuron array variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_list = []
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        row_list = []
        data_pt = NESTGPU_GetArrayVar(ctypes.c_int(i_node1), c_var_name)
        array_size = GetNeuronVarSize(i_node1, var_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def GetNeuronListArrayVar(node_list, var_name):
    "Get neuron array variable"
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    data_list = []
    for i_node in node_list:
        row_list = []
        data_pt = NESTGPU_GetArrayVar(ctypes.c_int(i_node), c_var_name)
        array_size = GetNeuronVarSize(i_node, var_name)
        for i in range(array_size):
            row_list.append(data_pt[i])
        data_list.append(row_list)
        
    ret = data_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetNeuronStatus(nodes, var_name):
    "Get neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    if type(nodes)==NodeSeq:
        if (IsNeuronScalParam(nodes.i0, var_name) |
            IsNeuronPortParam(nodes.i0, var_name)):
            ret = GetNeuronParam(nodes.i0, nodes.n, var_name)
        elif IsNeuronArrayParam(nodes.i0, var_name):
            ret = GetArrayParam(nodes.i0, nodes.n, var_name)
        elif (IsNeuronIntVar(nodes.i0, var_name)):
            ret = GetNeuronIntVar(nodes.i0, nodes.n, var_name)
        elif (IsNeuronScalVar(nodes.i0, var_name) |
              IsNeuronPortVar(nodes.i0, var_name)):
            ret = GetNeuronVar(nodes.i0, nodes.n, var_name)
        elif IsNeuronArrayVar(nodes.i0, var_name):
            ret = GetArrayVar(nodes.i0, nodes.n, var_name)
        elif IsNeuronGroupParam(nodes.i0, var_name):
            ret = GetNeuronStatus(nodes.ToList(), var_name)
            
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:
        if (IsNeuronScalParam(nodes[0], var_name) |
            IsNeuronPortParam(nodes[0], var_name)):
            ret = GetNeuronPtParam(nodes, var_name)
        elif IsNeuronArrayParam(nodes[0], var_name):
            ret = GetNeuronListArrayParam(nodes, var_name)
        elif (IsNeuronIntVar(nodes[0], var_name)):
            ret = GetNeuronPtIntVar(nodes, var_name)
        elif (IsNeuronScalVar(nodes[0], var_name) |
              IsNeuronPortVar(nodes[0], var_name)):
            ret = GetNeuronPtVar(nodes, var_name)
        elif IsNeuronArrayVar(nodes[0], var_name):
            ret = GetNeuronListArrayVar(nodes, var_name)
        elif IsNeuronGroupParam(nodes[0], var_name):
            ret = []
            for i_node in nodes:
                ret.append(GetNeuronGroupParam(i_node, var_name))
        else:
            raise ValueError("Unknown neuron variable or parameter")
    return ret


NESTGPU_GetNIntVar = _nestgpu.NESTGPU_GetNIntVar
NESTGPU_GetNIntVar.argtypes = (ctypes.c_int,)
NESTGPU_GetNIntVar.restype = ctypes.c_int
def GetNIntVar(i_node):
    "Get number of integer variables for a given node"
    ret = NESTGPU_GetNIntVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetNScalVar = _nestgpu.NESTGPU_GetNScalVar
NESTGPU_GetNScalVar.argtypes = (ctypes.c_int,)
NESTGPU_GetNScalVar.restype = ctypes.c_int
def GetNScalVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NESTGPU_GetNScalVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetIntVarNames = _nestgpu.NESTGPU_GetIntVarNames
NESTGPU_GetIntVarNames.argtypes = (ctypes.c_int,)
NESTGPU_GetIntVarNames.restype = ctypes.POINTER(c_char_p)
def GetIntVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNIntVar(i_node)
    var_name_pp = ctypes.cast(NESTGPU_GetIntVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list

NESTGPU_GetScalVarNames = _nestgpu.NESTGPU_GetScalVarNames
NESTGPU_GetScalVarNames.argtypes = (ctypes.c_int,)
NESTGPU_GetScalVarNames.restype = ctypes.POINTER(c_char_p)
def GetScalVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNScalVar(i_node)
    var_name_pp = ctypes.cast(NESTGPU_GetScalVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list

NESTGPU_GetNPortVar = _nestgpu.NESTGPU_GetNPortVar
NESTGPU_GetNPortVar.argtypes = (ctypes.c_int,)
NESTGPU_GetNPortVar.restype = ctypes.c_int
def GetNPortVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NESTGPU_GetNPortVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetPortVarNames = _nestgpu.NESTGPU_GetPortVarNames
NESTGPU_GetPortVarNames.argtypes = (ctypes.c_int,)
NESTGPU_GetPortVarNames.restype = ctypes.POINTER(c_char_p)
def GetPortVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNPortVar(i_node)
    var_name_pp = ctypes.cast(NESTGPU_GetPortVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list


NESTGPU_GetNScalParam = _nestgpu.NESTGPU_GetNScalParam
NESTGPU_GetNScalParam.argtypes = (ctypes.c_int,)
NESTGPU_GetNScalParam.restype = ctypes.c_int
def GetNScalParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NESTGPU_GetNScalParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetScalParamNames = _nestgpu.NESTGPU_GetScalParamNames
NESTGPU_GetScalParamNames.argtypes = (ctypes.c_int,)
NESTGPU_GetScalParamNames.restype = ctypes.POINTER(c_char_p)
def GetScalParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNScalParam(i_node)
    param_name_pp = ctypes.cast(NESTGPU_GetScalParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list

NESTGPU_GetNPortParam = _nestgpu.NESTGPU_GetNPortParam
NESTGPU_GetNPortParam.argtypes = (ctypes.c_int,)
NESTGPU_GetNPortParam.restype = ctypes.c_int
def GetNPortParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NESTGPU_GetNPortParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetPortParamNames = _nestgpu.NESTGPU_GetPortParamNames
NESTGPU_GetPortParamNames.argtypes = (ctypes.c_int,)
NESTGPU_GetPortParamNames.restype = ctypes.POINTER(c_char_p)
def GetPortParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNPortParam(i_node)
    param_name_pp = ctypes.cast(NESTGPU_GetPortParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NESTGPU_GetNArrayParam = _nestgpu.NESTGPU_GetNArrayParam
NESTGPU_GetNArrayParam.argtypes = (ctypes.c_int,)
NESTGPU_GetNArrayParam.restype = ctypes.c_int
def GetNArrayParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NESTGPU_GetNArrayParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetArrayParamNames = _nestgpu.NESTGPU_GetArrayParamNames
NESTGPU_GetArrayParamNames.argtypes = (ctypes.c_int,)
NESTGPU_GetArrayParamNames.restype = ctypes.POINTER(c_char_p)
def GetArrayParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNArrayParam(i_node)
    param_name_pp = ctypes.cast(NESTGPU_GetArrayParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NESTGPU_GetNGroupParam = _nestgpu.NESTGPU_GetNGroupParam
NESTGPU_GetNGroupParam.argtypes = (ctypes.c_int,)
NESTGPU_GetNGroupParam.restype = ctypes.c_int
def GetNGroupParam(i_node):
    "Get number of scalar parameters for a given node"
    ret = NESTGPU_GetNGroupParam(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetGroupParamNames = _nestgpu.NESTGPU_GetGroupParamNames
NESTGPU_GetGroupParamNames.argtypes = (ctypes.c_int,)
NESTGPU_GetGroupParamNames.restype = ctypes.POINTER(c_char_p)
def GetGroupParamNames(i_node):
    "Get list of scalar parameter names"
    n_param = GetNGroupParam(i_node)
    param_name_pp = ctypes.cast(NESTGPU_GetGroupParamNames(
        ctypes.c_int(i_node)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list

NESTGPU_GetNArrayVar = _nestgpu.NESTGPU_GetNArrayVar
NESTGPU_GetNArrayVar.argtypes = (ctypes.c_int,)
NESTGPU_GetNArrayVar.restype = ctypes.c_int
def GetNArrayVar(i_node):
    "Get number of scalar variables for a given node"
    ret = NESTGPU_GetNArrayVar(ctypes.c_int(i_node))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetArrayVarNames = _nestgpu.NESTGPU_GetArrayVarNames
NESTGPU_GetArrayVarNames.argtypes = (ctypes.c_int,)
NESTGPU_GetArrayVarNames.restype = ctypes.POINTER(c_char_p)
def GetArrayVarNames(i_node):
    "Get list of scalar variable names"
    n_var = GetNArrayVar(i_node)
    var_name_pp = ctypes.cast(NESTGPU_GetArrayVarNames(ctypes.c_int(i_node)),
                               ctypes.POINTER(c_char_p))
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name = ctypes.cast(var_name_p, ctypes.c_char_p).value
        var_name_list.append(to_def_str(var_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return var_name_list




def SetNeuronStatus(nodes, var_name, val):
    "Set neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    if (type(val)==dict):
        array_size = len(nodes)
        arr = DictToArray(val, array_size)
        for i in range(array_size):
            SetNeuronStatus([nodes[i]], var_name, arr[i])
        return
    
    c_var_name = ctypes.create_string_buffer(to_byte_str(var_name),
                                               len(var_name)+1)
    if type(nodes)==NodeSeq:
        if IsNeuronGroupParam(nodes.i0, var_name):
            SetNeuronGroupParam(nodes, var_name, val)
        elif IsNeuronScalParam(nodes.i0, var_name):
            SetNeuronScalParam(nodes.i0, nodes.n, var_name, val)
        elif (IsNeuronPortParam(nodes.i0, var_name) |
              IsNeuronArrayParam(nodes.i0, var_name)):
            SetNeuronArrayParam(nodes.i0, nodes.n, var_name, val)
        elif IsNeuronIntVar(nodes.i0, var_name):
            SetNeuronIntVar(nodes.i0, nodes.n, var_name, val)
        elif IsNeuronScalVar(nodes.i0, var_name):
            SetNeuronScalVar(nodes.i0, nodes.n, var_name, val)
        elif (IsNeuronPortVar(nodes.i0, var_name) |
              IsNeuronArrayVar(nodes.i0, var_name)):
            SetNeuronArrayVar(nodes.i0, nodes.n, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:        
        if IsNeuronScalParam(nodes[0], var_name):
            SetNeuronPtScalParam(nodes, var_name, val)
        elif (IsNeuronPortParam(nodes[0], var_name) |
              IsNeuronArrayParam(nodes[0], var_name)):
            SetNeuronPtArrayParam(nodes, var_name, val)
        elif IsNeuronIntVar(nodes[0], var_name):
            SetNeuronPtIntVar(nodes, var_name, val)
        elif IsNeuronScalVar(nodes[0], var_name):
            SetNeuronPtScalVar(nodes, var_name, val)
        elif (IsNeuronPortVar(nodes[0], var_name) |
              IsNeuronArrayVar(nodes[0], var_name)):
            SetNeuronPtArrayVar(nodes, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")


NESTGPU_Calibrate = _nestgpu.NESTGPU_Calibrate
NESTGPU_Calibrate.restype = ctypes.c_int
def Calibrate():
    "Calibrate simulation"
    ret = NESTGPU_Calibrate()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_Simulate = _nestgpu.NESTGPU_Simulate
NESTGPU_Simulate.restype = ctypes.c_int
def Simulate(sim_time=1000.0):
    "Simulate neural activity"
    SetSimTime(sim_time)
    ret = NESTGPU_Simulate()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_ConnectMpiInit = _nestgpu.NESTGPU_ConnectMpiInit
NESTGPU_ConnectMpiInit.argtypes = (ctypes.c_int, ctypes.POINTER(c_char_p))
NESTGPU_ConnectMpiInit.restype = ctypes.c_int
def ConnectMpiInit():
    "Initialize MPI connections"
    argc=len(sys.argv)
    array_char_pt_type = c_char_p * argc
    c_var_name_list=[]
    for i in range(argc):
        c_arg = ctypes.create_string_buffer(to_byte_str(sys.argv[i]), 100)
        c_var_name_list.append(c_arg)        
    ret = NESTGPU_ConnectMpiInit(ctypes.c_int(argc),
                                   array_char_pt_type(*c_var_name_list))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_MpiId = _nestgpu.NESTGPU_MpiId
NESTGPU_MpiId.restype = ctypes.c_int
def MpiId():
    "Get MPI Id"
    ret = NESTGPU_MpiId()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def Rank():
    "Get MPI rank"
    return MpiId()

NESTGPU_MpiNp = _nestgpu.NESTGPU_MpiNp
NESTGPU_MpiNp.restype = ctypes.c_int
def MpiNp():
    "Get MPI Np"
    ret = NESTGPU_MpiNp()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_ProcMaster = _nestgpu.NESTGPU_ProcMaster
NESTGPU_ProcMaster.restype = ctypes.c_int
def ProcMaster():
    "Get MPI ProcMaster"
    ret = NESTGPU_ProcMaster()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_MpiFinalize = _nestgpu.NESTGPU_MpiFinalize
NESTGPU_MpiFinalize.restype = ctypes.c_int
def MpiFinalize():
    "Finalize MPI"
    ret = NESTGPU_MpiFinalize()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_RandomInt = _nestgpu.NESTGPU_RandomInt
NESTGPU_RandomInt.argtypes = (ctypes.c_size_t,)
NESTGPU_RandomInt.restype = ctypes.POINTER(ctypes.c_uint)
def RandomInt(n):
    "Generate n random integers in CUDA memory"
    ret = NESTGPU_RandomInt(ctypes.c_size_t(n))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_RandomUniform = _nestgpu.NESTGPU_RandomUniform
NESTGPU_RandomUniform.argtypes = (ctypes.c_size_t,)
NESTGPU_RandomUniform.restype = c_float_p
def RandomUniform(n):
    "Generate n random floats with uniform distribution in (0,1) in CUDA memory"
    ret = NESTGPU_RandomUniform(ctypes.c_size_t(n))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_RandomNormal = _nestgpu.NESTGPU_RandomNormal
NESTGPU_RandomNormal.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float)
NESTGPU_RandomNormal.restype = c_float_p
def RandomNormal(n, mean, stddev):
    "Generate n random floats with normal distribution in CUDA memory"
    ret = NESTGPU_RandomNormal(ctypes.c_size_t(n), ctypes.c_float(mean),
                                 ctypes.c_float(stddev))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_RandomNormalClipped = _nestgpu.NESTGPU_RandomNormalClipped
NESTGPU_RandomNormalClipped.argtypes = (ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                          ctypes.c_float, ctypes.c_float)
NESTGPU_RandomNormalClipped.restype = c_float_p
def RandomNormalClipped(n, mean, stddev, vmin, vmax, vstep=0):
    "Generate n random floats with normal clipped distribution in CUDA memory"
    ret = NESTGPU_RandomNormalClipped(ctypes.c_size_t(n),
                                        ctypes.c_float(mean),
                                        ctypes.c_float(stddev),
                                        ctypes.c_float(vmin),
                                        ctypes.c_float(vmax),
                                        ctypes.c_float(vstep))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_Connect = _nestgpu.NESTGPU_Connect
NESTGPU_Connect.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_ubyte, ctypes.c_float, ctypes.c_float)
NESTGPU_Connect.restype = ctypes.c_int
def SingleConnect(i_source_node, i_target_node, i_port, weight, delay):
    "Connect two nodes"
    ret = NESTGPU_Connect(ctypes.c_int(i_source_node),
                            ctypes.c_int(i_target_node),
                            ctypes.c_ubyte(i_port), ctypes.c_float(weight),
                            ctypes.c_float(delay))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_ConnSpecInit = _nestgpu.NESTGPU_ConnSpecInit
NESTGPU_ConnSpecInit.restype = ctypes.c_int
def ConnSpecInit():
    "Initialize connection rules specification"
    ret = NESTGPU_ConnSpecInit()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetConnSpecParam = _nestgpu.NESTGPU_SetConnSpecParam
NESTGPU_SetConnSpecParam.argtypes = (c_char_p, ctypes.c_int)
NESTGPU_SetConnSpecParam.restype = ctypes.c_int
def SetConnSpecParam(param_name, val):
    "Set connection parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NESTGPU_SetConnSpecParam(c_param_name, ctypes.c_int(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_ConnSpecIsParam = _nestgpu.NESTGPU_ConnSpecIsParam
NESTGPU_ConnSpecIsParam.argtypes = (c_char_p,)
NESTGPU_ConnSpecIsParam.restype = ctypes.c_int
def ConnSpecIsParam(param_name):
    "Check name of connection parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NESTGPU_ConnSpecIsParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SynSpecInit = _nestgpu.NESTGPU_SynSpecInit
NESTGPU_SynSpecInit.restype = ctypes.c_int
def SynSpecInit():
    "Initializa synapse specification"
    ret = NESTGPU_SynSpecInit()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_SetSynSpecIntParam = _nestgpu.NESTGPU_SetSynSpecIntParam
NESTGPU_SetSynSpecIntParam.argtypes = (c_char_p, ctypes.c_int)
NESTGPU_SetSynSpecIntParam.restype = ctypes.c_int
def SetSynSpecIntParam(param_name, val):
    "Set synapse int parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NESTGPU_SetSynSpecIntParam(c_param_name, ctypes.c_int(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_SetSynSpecFloatParam = _nestgpu.NESTGPU_SetSynSpecFloatParam
NESTGPU_SetSynSpecFloatParam.argtypes = (c_char_p, ctypes.c_float)
NESTGPU_SetSynSpecFloatParam.restype = ctypes.c_int
def SetSynSpecFloatParam(param_name, val):
    "Set synapse float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = NESTGPU_SetSynSpecFloatParam(c_param_name, ctypes.c_float(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_SetSynSpecFloatPtParam = _nestgpu.NESTGPU_SetSynSpecFloatPtParam
NESTGPU_SetSynSpecFloatPtParam.argtypes = (c_char_p, ctypes.c_void_p)
NESTGPU_SetSynSpecFloatPtParam.restype = ctypes.c_int
def SetSynSpecFloatPtParam(param_name, arr):
    "Set synapse pointer to float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    if (type(arr) is list)  | (type(arr) is tuple):
        arr = (ctypes.c_float * len(arr))(*arr)
    arr_pt = ctypes.cast(arr, ctypes.c_void_p)
    ret = NESTGPU_SetSynSpecFloatPtParam(c_param_name, arr_pt)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SynSpecIsIntParam = _nestgpu.NESTGPU_SynSpecIsIntParam
NESTGPU_SynSpecIsIntParam.argtypes = (c_char_p,)
NESTGPU_SynSpecIsIntParam.restype = ctypes.c_int
def SynSpecIsIntParam(param_name):
    "Check name of synapse int parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NESTGPU_SynSpecIsIntParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SynSpecIsFloatParam = _nestgpu.NESTGPU_SynSpecIsFloatParam
NESTGPU_SynSpecIsFloatParam.argtypes = (c_char_p,)
NESTGPU_SynSpecIsFloatParam.restype = ctypes.c_int
def SynSpecIsFloatParam(param_name):
    "Check name of synapse float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NESTGPU_SynSpecIsFloatParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SynSpecIsFloatPtParam = _nestgpu.NESTGPU_SynSpecIsFloatPtParam
NESTGPU_SynSpecIsFloatPtParam.argtypes = (c_char_p,)
NESTGPU_SynSpecIsFloatPtParam.restype = ctypes.c_int
def SynSpecIsFloatPtParam(param_name):
    "Check name of synapse pointer to float parameter"
    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name), len(param_name)+1)
    ret = (NESTGPU_SynSpecIsFloatPtParam(c_param_name) != 0)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def DictToArray(param_dict, array_size):
    dist_name = None
    arr = None
    low = -1.0e35
    high = 1.0e35
    mu = None
    sigma = None
    vstep = 0
    
    for param_name in param_dict:
        pval = param_dict[param_name]
        if param_name=="array":
            dist_name = "array"
            arr = pval
        elif param_name=="distribution":
            dist_name = pval
        elif param_name=="low":
            low = pval
        elif param_name=="high":
            high = pval
        elif param_name=="mu":
            mu = pval
        elif param_name=="sigma":
            sigma = pval
        elif param_name=="step":
            vstep = pval
        else:
            raise ValueError("Unknown parameter name in dictionary")

    if dist_name=="array":
        if (type(arr) is list) | (type(arr) is tuple):
            if len(arr) != array_size:
                raise ValueError("Wrong array size.")
            arr = (ctypes.c_float * len(arr))(*arr)
            #array_pt = ctypes.cast(arr, ctypes.c_void_p)
            #return array_pt
        return arr
    elif dist_name=="normal":
        return RandomNormal(array_size, mu, sigma)
    elif dist_name=="normal_clipped":
        return RandomNormalClipped(array_size, mu, sigma, low, high, vstep)
    else:
        raise ValueError("Unknown distribution")


def RuleArraySize(conn_dict, source, target):
    if conn_dict["rule"]=="one_to_one":
        array_size = len(source)
    elif conn_dict["rule"]=="all_to_all":
        array_size = len(source)*len(target)
    elif conn_dict["rule"]=="fixed_total_number":
        array_size = conn_dict["total_num"]
    elif conn_dict["rule"]=="fixed_indegree":
        array_size = len(target)*conn_dict["indegree"]
    elif conn_dict["rule"]=="fixed_outdegree":
        array_size = len(source)*conn_dict["outdegree"]
    else:
        raise ValueError("Unknown number of connections for this rule")
    return array_size


def SetSynParamFromArray(param_name, par_dict, array_size):
    arr_param_name = param_name + "_array"
    if (not SynSpecIsFloatPtParam(arr_param_name)):
        raise ValueError("Synapse parameter cannot be set by"
                         " arrays or distributions")
    arr = DictToArray(par_dict, array_size)
        
    array_pt = ctypes.cast(arr, ctypes.c_void_p)
    SetSynSpecFloatPtParam(arr_param_name, array_pt)

    
NESTGPU_ConnectSeqSeq = _nestgpu.NESTGPU_ConnectSeqSeq
NESTGPU_ConnectSeqSeq.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int)
NESTGPU_ConnectSeqSeq.restype = ctypes.c_int

NESTGPU_ConnectSeqGroup = _nestgpu.NESTGPU_ConnectSeqGroup
NESTGPU_ConnectSeqGroup.argtypes = (ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_int)
NESTGPU_ConnectSeqGroup.restype = ctypes.c_int

NESTGPU_ConnectGroupSeq = _nestgpu.NESTGPU_ConnectGroupSeq
NESTGPU_ConnectGroupSeq.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int)
NESTGPU_ConnectGroupSeq.restype = ctypes.c_int

NESTGPU_ConnectGroupGroup = _nestgpu.NESTGPU_ConnectGroupGroup
NESTGPU_ConnectGroupGroup.argtypes = (ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int)
NESTGPU_ConnectGroupGroup.restype = ctypes.c_int

def Connect(source, target, conn_dict, syn_dict): 
    "Connect two node groups"
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")

    gc.disable()
    ConnSpecInit()
    SynSpecInit()
    for param_name in conn_dict:
        if param_name=="rule":
            for i_rule in range(len(conn_rule_name)):
                if conn_dict[param_name]==conn_rule_name[i_rule]:
                    break
            if i_rule < len(conn_rule_name):
                SetConnSpecParam(param_name, i_rule)
            else:
                raise ValueError("Unknown connection rule")
        elif ConnSpecIsParam(param_name):
            SetConnSpecParam(param_name, conn_dict[param_name])
        else:
            raise ValueError("Unknown connection parameter")
    
    array_size = RuleArraySize(conn_dict, source, target)
    
    for param_name in syn_dict:
        if SynSpecIsIntParam(param_name):
            val = syn_dict[param_name]
            if ((param_name=="synapse_group") & (type(val)==SynGroup)):
                val = val.i_syn_group
            SetSynSpecIntParam(param_name, val)
        elif SynSpecIsFloatParam(param_name):
            fpar = syn_dict[param_name]
            if (type(fpar)==dict):
                SetSynParamFromArray(param_name, fpar, array_size)
            else:
                SetSynSpecFloatParam(param_name, fpar)

        elif SynSpecIsFloatPtParam(param_name):
            SetSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = NESTGPU_ConnectSeqSeq(source.i0, source.n, target.i0, target.n)
    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = NESTGPU_ConnectSeqGroup(source.i0, source.n, target_arr_pt,
                                            len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = NESTGPU_ConnectGroupSeq(source_arr_pt, len(source),
                                            target.i0, target.n)
        else:
            ret = NESTGPU_ConnectGroupGroup(source_arr_pt, len(source),
                                              target_arr_pt, len(target))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    gc.enable()
    return ret


NESTGPU_RemoteConnectSeqSeq = _nestgpu.NESTGPU_RemoteConnectSeqSeq
NESTGPU_RemoteConnectSeqSeq.argtypes = (ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int)
NESTGPU_RemoteConnectSeqSeq.restype = ctypes.c_int

NESTGPU_RemoteConnectSeqGroup = _nestgpu.NESTGPU_RemoteConnectSeqGroup
NESTGPU_RemoteConnectSeqGroup.argtypes = (ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_int)
NESTGPU_RemoteConnectSeqGroup.restype = ctypes.c_int

NESTGPU_RemoteConnectGroupSeq = _nestgpu.NESTGPU_RemoteConnectGroupSeq
NESTGPU_RemoteConnectGroupSeq.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int)
NESTGPU_RemoteConnectGroupSeq.restype = ctypes.c_int

NESTGPU_RemoteConnectGroupGroup = _nestgpu.NESTGPU_RemoteConnectGroupGroup
NESTGPU_RemoteConnectGroupGroup.argtypes = (ctypes.c_int, ctypes.c_void_p,
                                              ctypes.c_int, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_int)
NESTGPU_RemoteConnectGroupGroup.restype = ctypes.c_int

def RemoteConnect(i_source_host, source, i_target_host, target,
                  conn_dict, syn_dict): 
    "Connect two node groups of differen mpi hosts"
    if (type(i_source_host)!=int) | (type(i_target_host)!=int):
        raise ValueError("Error in host index")
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")
        
    ConnSpecInit()
    SynSpecInit()
    for param_name in conn_dict:
        if param_name=="rule":
            for i_rule in range(len(conn_rule_name)):
                if conn_dict[param_name]==conn_rule_name[i_rule]:
                    break
            if i_rule < len(conn_rule_name):
                SetConnSpecParam(param_name, i_rule)
            else:
                raise ValueError("Unknown connection rule")
                
        elif ConnSpecIsParam(param_name):
            SetConnSpecParam(param_name, conn_dict[param_name])
        else:
            raise ValueError("Unknown connection parameter")
        
    array_size = RuleArraySize(conn_dict, source, target)    
        
    for param_name in syn_dict:
        if SynSpecIsIntParam(param_name):
            SetSynSpecIntParam(param_name, syn_dict[param_name])
        elif SynSpecIsFloatParam(param_name):
            fpar = syn_dict[param_name]
            if (type(fpar)==dict):
                SetSynParamFromArray(param_name, fpar, array_size)
            else:
                SetSynSpecFloatParam(param_name, fpar)
                
        elif SynSpecIsFloatPtParam(param_name):
            SetSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = NESTGPU_RemoteConnectSeqSeq(i_source_host, source.i0, source.n,
                                            i_target_host, target.i0, target.n)

    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = NESTGPU_RemoteConnectSeqGroup(i_source_host, source.i0,
                                                  source.n, i_target_host,
                                                  target_arr_pt, len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = NESTGPU_RemoteConnectGroupSeq(i_source_host, source_arr_pt,
                                                  len(source),
                                                  i_target_host, target.i0,
                                                  target.n)
        else:
            ret = NESTGPU_RemoteConnectGroupGroup(i_source_host,
                                                    source_arr_pt,
                                                    len(source),
                                                    i_target_host,
                                                    target_arr_pt,
                                                    len(target))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


def SetStatus(gen_object, params, val=None):
    "Set neuron or synapse group parameters or variables using dictionaries"

    if type(gen_object)==RemoteNodeSeq:
        if gen_object.i_host==MpiId():
            SetStatus(gen_object.node_seq, params, val)
        return
    
    gc.disable()
    if type(gen_object)==SynGroup:
        ret = SetSynGroupStatus(gen_object, params, val)
        gc.enable()
        return ret
    nodes = gen_object    
    if val != None:
         SetNeuronStatus(nodes, params, val)
    elif type(params)==dict:
        for param_name in params:
            SetNeuronStatus(nodes, param_name, params[param_name])
    elif (type(params)==list)  | (type(params) is tuple):
        if len(params) != len(nodes):
            raise ValueError("List should have the same size as nodes")
        for param_dict in params:
            if type(param_dict)!=dict:
                raise ValueError("Type of list elements should be dict")
            for param_name in param_dict:
                SetNeuronStatus(nodes, param_name, param_dict[param_name])
    else:
        raise ValueError("Wrong argument in SetStatus")
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    gc.enable()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

NESTGPU_GetSeqSeqConnections = _nestgpu.NESTGPU_GetSeqSeqConnections
NESTGPU_GetSeqSeqConnections.argtypes = (ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, c_int_p)
NESTGPU_GetSeqSeqConnections.restype = c_int_p

NESTGPU_GetSeqGroupConnections = _nestgpu.NESTGPU_GetSeqGroupConnections
NESTGPU_GetSeqGroupConnections.argtypes = (ctypes.c_int, ctypes.c_int,
                                             c_void_p, ctypes.c_int,
                                             ctypes.c_int, c_int_p)
NESTGPU_GetSeqGroupConnections.restype = c_int_p

NESTGPU_GetGroupSeqConnections = _nestgpu.NESTGPU_GetGroupSeqConnections
NESTGPU_GetGroupSeqConnections.argtypes = (c_void_p, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, c_int_p)
NESTGPU_GetGroupSeqConnections.restype = c_int_p

NESTGPU_GetGroupGroupConnections = _nestgpu.NESTGPU_GetGroupGroupConnections
NESTGPU_GetGroupGroupConnections.argtypes = (c_void_p, ctypes.c_int,
                                               c_void_p, ctypes.c_int,
                                               ctypes.c_int, c_int_p)
NESTGPU_GetGroupGroupConnections.restype = c_int_p

def GetConnections(source=None, target=None, syn_group=-1): 
    "Get connections between two node groups"
    if source==None:
        source = NodeSeq(None)
    if target==None:
        target = NodeSeq(None)
    if (type(source)==int):
        source = [source]
    if (type(target)==int):
        target = [target]
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")
    
    n_conn = ctypes.c_int(0)
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        conn_arr = NESTGPU_GetSeqSeqConnections(source.i0, source.n,
                                                  target.i0, target.n,
                                                  syn_group,
                                                  ctypes.byref(n_conn))
    else:
        if type(source)!=NodeSeq:
            source_arr = (ctypes.c_int * len(source))(*source) 
            source_arr_pt = ctypes.cast(source_arr, ctypes.c_void_p)    
        if type(target)!=NodeSeq:
            target_arr = (ctypes.c_int * len(target))(*target) 
            target_arr_pt = ctypes.cast(target_arr, ctypes.c_void_p)    
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            conn_arr = NESTGPU_GetSeqGroupConnections(source.i0, source.n,
                                                        target_arr_pt,
                                                        len(target),
                                                        syn_group,
                                                        ctypes.byref(n_conn))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            conn_arr = NESTGPU_GetGroupSeqConnections(source_arr_pt,
                                                        len(source),
                                                        target.i0, target.n,
                                                        syn_group,
                                                        ctypes.byref(n_conn))
        else:
            conn_arr = NESTGPU_GetGroupGroupConnections(source_arr_pt,
                                                          len(source),
                                                          target_arr_pt,
                                                          len(target),
                                                          syn_group,
                                                          ctypes.byref(n_conn))

    conn_list = []
    for i_conn in range(n_conn.value):
        conn_id = ConnectionId(conn_arr[i_conn*3], conn_arr[i_conn*3 + 1],
                   conn_arr[i_conn*3 + 2])
        conn_list.append(conn_id)
        
    ret = conn_list

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

 
NESTGPU_GetConnectionStatus = _nestgpu.NESTGPU_GetConnectionStatus
NESTGPU_GetConnectionStatus.argtypes = (ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, c_int_p,
                                         c_char_p, c_char_p,
                                         c_float_p, c_float_p)
NESTGPU_GetConnectionStatus.restype = ctypes.c_int

def GetConnectionStatus(conn_id):
    i_source = conn_id.i_source
    i_group = conn_id.i_group
    i_conn = conn_id.i_conn
    
    i_target = ctypes.c_int(0)
    i_port = ctypes.c_char()
    i_syn = ctypes.c_char()
    delay = ctypes.c_float(0.0)
    weight = ctypes.c_float(0.0)

    NESTGPU_GetConnectionStatus(i_source, i_group, i_conn,
                                  ctypes.byref(i_target),
                                  ctypes.byref(i_port),
                                  ctypes.byref(i_syn),
                                  ctypes.byref(delay),
                                  ctypes.byref(weight))
    i_target = i_target.value
    i_port = ord(i_port.value)
    i_syn = ord(i_syn.value)
    delay = delay.value
    weight = weight.value
    conn_status_dict = {"source":i_source, "target":i_target, "port":i_port,
                        "syn":i_syn, "delay":delay, "weight":weight}

    return conn_status_dict


def GetStatus(gen_object, var_key=None):
    "Get neuron group, connection or synapse group status"
    if type(gen_object)==SynGroup:
        return GetSynGroupStatus(gen_object, var_key)
    
    if type(gen_object)==NodeSeq:
        gen_object = gen_object.ToList()
    if (type(gen_object)==list) | (type(gen_object)==tuple):
        status_list = []
        for gen_elem in gen_object:
            elem_dict = GetStatus(gen_elem, var_key)
            status_list.append(elem_dict)
        return status_list
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetStatus(gen_object, var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        if (type(gen_object)==ConnectionId):
            status_dict = GetConnectionStatus(gen_object)
        elif (type(gen_object)==int):
            i_node = gen_object
            status_dict = {}
            name_list = GetIntVarNames(i_node) \
                        + GetScalVarNames(i_node) + GetScalParamNames(i_node) \
                        + GetPortVarNames(i_node) + GetPortParamNames(i_node) \
                        + GetArrayVarNames(i_node) \
                        + GetArrayParamNames(i_node) \
                        + GetGroupParamNames(i_node)
            for var_name in name_list:
                val = GetStatus(i_node, var_name)
                status_dict[var_name] = val
        else:
            raise ValueError("Unknown object type in GetStatus")
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
        if (type(gen_object)==ConnectionId):
            status_dict = GetConnectionStatus(gen_object)
            return status_dict[var_key]
        elif (type(gen_object)==int):
            i_node = gen_object
            return GetNeuronStatus([i_node], var_key)[0]
        else:
            raise ValueError("Unknown object type in GetStatus")
        
    else:
        raise ValueError("Unknown key type in GetStatus", type(var_key))



NESTGPU_CreateSynGroup = _nestgpu.NESTGPU_CreateSynGroup
NESTGPU_CreateSynGroup.argtypes = (c_char_p,)
NESTGPU_CreateSynGroup.restype = ctypes.c_int
def CreateSynGroup(model_name, status_dict=None):
    "Create a synapse group"
    if (type(status_dict)==dict):
        syn_group = CreateSynGroup(model_name)
        SetStatus(syn_group, status_dict)
        return syn_group
    elif status_dict!=None:
        raise ValueError("Wrong argument in CreateSynGroup")

    c_model_name = ctypes.create_string_buffer(to_byte_str(model_name), \
                                               len(model_name)+1)
    i_syn_group = NESTGPU_CreateSynGroup(c_model_name) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return SynGroup(i_syn_group)

  
NESTGPU_GetSynGroupNParam = _nestgpu.NESTGPU_GetSynGroupNParam
NESTGPU_GetSynGroupNParam.argtypes = (ctypes.c_int,)
NESTGPU_GetSynGroupNParam.restype = ctypes.c_int
def GetSynGroupNParam(syn_group):
    "Get number of synapse parameters for a given synapse group"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupNParam")
    i_syn_group = syn_group.i_syn_group
    
    ret = NESTGPU_GetSynGroupNParam(ctypes.c_int(i_syn_group))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NESTGPU_GetSynGroupParamNames = _nestgpu.NESTGPU_GetSynGroupParamNames
NESTGPU_GetSynGroupParamNames.argtypes = (ctypes.c_int,)
NESTGPU_GetSynGroupParamNames.restype = ctypes.POINTER(c_char_p)
def GetSynGroupParamNames(syn_group):
    "Get list of synapse group parameter names"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupParamNames")
    i_syn_group = syn_group.i_syn_group

    n_param = GetSynGroupNParam(syn_group)
    param_name_pp = ctypes.cast(NESTGPU_GetSynGroupParamNames(
        ctypes.c_int(i_syn_group)), ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NESTGPU_IsSynGroupParam = _nestgpu.NESTGPU_IsSynGroupParam
NESTGPU_IsSynGroupParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_IsSynGroupParam.restype = ctypes.c_int
def IsSynGroupParam(syn_group, param_name):
    "Check name of synapse group parameter"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in IsSynGroupParam")
    i_syn_group = syn_group.i_syn_group

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NESTGPU_IsSynGroupParam(ctypes.c_int(i_syn_group), \
                                     c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

    
NESTGPU_GetSynGroupParam = _nestgpu.NESTGPU_GetSynGroupParam
NESTGPU_GetSynGroupParam.argtypes = (ctypes.c_int, c_char_p)
NESTGPU_GetSynGroupParam.restype = ctypes.c_float
def GetSynGroupParam(syn_group, param_name):
    "Get synapse group parameter value"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupParam")
    i_syn_group = syn_group.i_syn_group

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)

    ret = NESTGPU_GetSynGroupParam(ctypes.c_int(i_syn_group),
                                         c_param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NESTGPU_SetSynGroupParam = _nestgpu.NESTGPU_SetSynGroupParam
NESTGPU_SetSynGroupParam.argtypes = (ctypes.c_int, c_char_p,
                                       ctypes.c_float)
NESTGPU_SetSynGroupParam.restype = ctypes.c_int
def SetSynGroupParam(syn_group, param_name, val):
    "Set synapse group parameter value"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in SetSynGroupParam")
    i_syn_group = syn_group.i_syn_group

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NESTGPU_SetSynGroupParam(ctypes.c_int(i_syn_group),
                                         c_param_name, ctypes.c_float(val))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetSynGroupStatus(syn_group, var_key=None):
    "Get synapse group status"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupStatus")
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetSynGroupStatus(syn_group, var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        status_dict = {}
        name_list = GetSynGroupParamNames(syn_group)
        for param_name in name_list:
            val = GetSynGroupStatus(syn_group, param_name)
            status_dict[param_name] = val
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
            return GetSynGroupParam(syn_group, var_key)        
    else:
        raise ValueError("Unknown key type in GetSynGroupStatus", type(var_key))

def SetSynGroupStatus(syn_group, params, val=None):
    "Set synapse group parameters using dictionaries"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in SetSynGroupStatus")
    if ((type(params)==dict) & (val==None)):
        for param_name in params:
            SetSynGroupStatus(syn_group, param_name, params[param_name])
    elif (type(params)==str):
            return SetSynGroupParam(syn_group, params, val)        
    else:
        raise ValueError("Wrong argument in SetSynGroupStatus")       
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())


NESTGPU_ActivateSpikeCount = _nestgpu.NESTGPU_ActivateSpikeCount
NESTGPU_ActivateSpikeCount.argtypes = (ctypes.c_int, ctypes.c_int)
NESTGPU_ActivateSpikeCount.restype = ctypes.c_int
def ActivateSpikeCount(nodes):
    "Activate spike count for node group"
    if type(nodes)!=NodeSeq:
        raise ValueError("Argument type of ActivateSpikeCount must be NodeSeq")

    ret = NESTGPU_ActivateSpikeCount(ctypes.c_int(nodes.i0),
                                       ctypes.c_int(nodes.n))

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_ActivateRecSpikeTimes = _nestgpu.NESTGPU_ActivateRecSpikeTimes
NESTGPU_ActivateRecSpikeTimes.argtypes = (ctypes.c_int, ctypes.c_int, \
                                            ctypes.c_int)
NESTGPU_ActivateRecSpikeTimes.restype = ctypes.c_int
def ActivateRecSpikeTimes(nodes, max_n_rec_spike_times):
    "Activate spike time recording for node group"
    if type(nodes)!=NodeSeq:
        raise ValueError("Argument type of ActivateRecSpikeTimes must be NodeSeq")

    ret = NESTGPU_ActivateRecSpikeTimes(ctypes.c_int(nodes.i0),
                                          ctypes.c_int(nodes.n),
                                          ctypes.c_int(max_n_rec_spike_times))

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_SetRecSpikeTimesStep = _nestgpu.NESTGPU_SetRecSpikeTimesStep
NESTGPU_SetRecSpikeTimesStep.argtypes = (ctypes.c_int, ctypes.c_int, \
                                         ctypes.c_int)
NESTGPU_SetRecSpikeTimesStep.restype = ctypes.c_int
def SetRecSpikeTimesStep(nodes, rec_spike_times_step):
    "Setp number of time steps for buffering spike time recording"
    if type(nodes)!=NodeSeq:
        raise ValueError("Argument type of SetRecSpikeTimesStep must be NodeSeq")

    ret = NESTGPU_SetRecSpikeTimesStep(ctypes.c_int(nodes.i0),
                                       ctypes.c_int(nodes.n),
                                       ctypes.c_int(rec_spike_times_step))

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNRecSpikeTimes = _nestgpu.NESTGPU_GetNRecSpikeTimes
NESTGPU_GetNRecSpikeTimes.argtypes = (ctypes.c_int,)
NESTGPU_GetNRecSpikeTimes.restype = ctypes.c_int
def GetNRecSpikeTimes(i_node):
    "Get number of recorded spike times for node"

    ret = NESTGPU_GetNRecSpikeTimes(ctypes.c_int(i_node))

    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetRecSpikeTimes = _nestgpu.NESTGPU_GetRecSpikeTimes
NESTGPU_GetRecSpikeTimes.argtypes = (ctypes.c_int, ctypes.c_int, c_int_pp, c_float_ppp)
NESTGPU_GetRecSpikeTimes.restype = ctypes.c_int

def GetRecSpikeTimes(nodes):
    "Get recorded spike times for node group"
    if type(nodes)!=NodeSeq:
        raise ValueError("First argument type of GetRecSpikeTimes must be NodeSeq")

    n_spike_times = (c_int_p * 1)()
    n_spike_times_pt = ctypes.cast(n_spike_times, c_int_pp)    
    spike_times = (c_float_pp * 1)()
    spike_times_pt = ctypes.cast(spike_times, c_float_ppp)    

    
    spike_time_list = []
    ret1 = NESTGPU_GetRecSpikeTimes(ctypes.c_int(nodes.i0), ctypes.c_int(nodes.n),
                                    n_spike_times_pt, spike_times_pt)
    for i_n in range(nodes.n):
        spike_time_list.append([])
        n_spike = n_spike_times_pt[0][i_n]
        for i_spike in range(n_spike):
            spike_time_list[i_n].append(spike_times_pt[0][i_n][i_spike])
        
    ret = spike_time_list
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_SetNeuronGroupParam = _nestgpu.NESTGPU_SetNeuronGroupParam
NESTGPU_SetNeuronGroupParam.argtypes = (ctypes.c_int, ctypes.c_int,
                                          c_char_p, ctypes.c_float)
NESTGPU_SetNeuronGroupParam.restype = ctypes.c_int
def SetNeuronGroupParam(nodes, param_name, val):
    "Set neuron group parameter value"
    if type(nodes)!=NodeSeq:
        raise ValueError("Wrong argument type in SetNeuronGroupParam")

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NESTGPU_SetNeuronGroupParam(ctypes.c_int(nodes.i0),
                                        ctypes.c_int(nodes.n),
                                        c_param_name, ctypes.c_float(val))
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNBoolParam = _nestgpu.NESTGPU_GetNBoolParam
NESTGPU_GetNBoolParam.restype = ctypes.c_int
def GetNBoolParam():
    "Get number of kernel boolean parameters"
    
    ret = NESTGPU_GetNBoolParam()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetBoolParamNames = _nestgpu.NESTGPU_GetBoolParamNames
NESTGPU_GetBoolParamNames.restype = ctypes.POINTER(c_char_p)
def GetBoolParamNames():
    "Get list of kernel boolean parameter names"

    n_param = GetNBoolParam()
    param_name_pp = ctypes.cast(NESTGPU_GetBoolParamNames(),
                                ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NESTGPU_IsBoolParam = _nestgpu.NESTGPU_IsBoolParam
NESTGPU_IsBoolParam.argtypes = (c_char_p,)
NESTGPU_IsBoolParam.restype = ctypes.c_int
def IsBoolParam(param_name):
    "Check name of kernel boolean parameter"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NESTGPU_IsBoolParam(c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

    
NESTGPU_GetBoolParam = _nestgpu.NESTGPU_GetBoolParam
NESTGPU_GetBoolParam.argtypes = (c_char_p,)
NESTGPU_GetBoolParam.restype = ctypes.c_bool
def GetBoolParam(param_name):
    "Get kernel boolean parameter value"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)

    ret = NESTGPU_GetBoolParam(c_param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NESTGPU_SetBoolParam = _nestgpu.NESTGPU_SetBoolParam
NESTGPU_SetBoolParam.argtypes = (c_char_p, ctypes.c_bool)
NESTGPU_SetBoolParam.restype = ctypes.c_int
def SetBoolParam(param_name, val):
    "Set kernel boolean parameter value"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NESTGPU_SetBoolParam(c_param_name, ctypes.c_bool(val))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

NESTGPU_GetNFloatParam = _nestgpu.NESTGPU_GetNFloatParam
NESTGPU_GetNFloatParam.restype = ctypes.c_int
def GetNFloatParam():
    "Get number of kernel float parameters"
    
    ret = NESTGPU_GetNFloatParam()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetFloatParamNames = _nestgpu.NESTGPU_GetFloatParamNames
NESTGPU_GetFloatParamNames.restype = ctypes.POINTER(c_char_p)
def GetFloatParamNames():
    "Get list of kernel float parameter names"

    n_param = GetNFloatParam()
    param_name_pp = ctypes.cast(NESTGPU_GetFloatParamNames(),
                                ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NESTGPU_IsFloatParam = _nestgpu.NESTGPU_IsFloatParam
NESTGPU_IsFloatParam.argtypes = (c_char_p,)
NESTGPU_IsFloatParam.restype = ctypes.c_int
def IsFloatParam(param_name):
    "Check name of kernel float parameter"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NESTGPU_IsFloatParam(c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

    
NESTGPU_GetFloatParam = _nestgpu.NESTGPU_GetFloatParam
NESTGPU_GetFloatParam.argtypes = (c_char_p,)
NESTGPU_GetFloatParam.restype = ctypes.c_float
def GetFloatParam(param_name):
    "Get kernel float parameter value"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)

    ret = NESTGPU_GetFloatParam(c_param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NESTGPU_SetFloatParam = _nestgpu.NESTGPU_SetFloatParam
NESTGPU_SetFloatParam.argtypes = (c_char_p, ctypes.c_float)
NESTGPU_SetFloatParam.restype = ctypes.c_int
def SetFloatParam(param_name, val):
    "Set kernel float parameter value"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NESTGPU_SetFloatParam(c_param_name, ctypes.c_float(val))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetNIntParam = _nestgpu.NESTGPU_GetNIntParam
NESTGPU_GetNIntParam.restype = ctypes.c_int
def GetNIntParam():
    "Get number of kernel int parameters"
    
    ret = NESTGPU_GetNIntParam()
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret


NESTGPU_GetIntParamNames = _nestgpu.NESTGPU_GetIntParamNames
NESTGPU_GetIntParamNames.restype = ctypes.POINTER(c_char_p)
def GetIntParamNames():
    "Get list of kernel int parameter names"

    n_param = GetNIntParam()
    param_name_pp = ctypes.cast(NESTGPU_GetIntParamNames(),
                                ctypes.POINTER(c_char_p))
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name = ctypes.cast(param_name_p, ctypes.c_char_p).value
        param_name_list.append(to_def_str(param_name))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return param_name_list


NESTGPU_IsIntParam = _nestgpu.NESTGPU_IsIntParam
NESTGPU_IsIntParam.argtypes = (c_char_p,)
NESTGPU_IsIntParam.restype = ctypes.c_int
def IsIntParam(param_name):
    "Check name of kernel int parameter"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = (NESTGPU_IsIntParam(c_param_name)!=0) 
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

    
NESTGPU_GetIntParam = _nestgpu.NESTGPU_GetIntParam
NESTGPU_GetIntParam.argtypes = (c_char_p,)
NESTGPU_GetIntParam.restype = ctypes.c_int
def GetIntParam(param_name):
    "Get kernel int parameter value"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)

    ret = NESTGPU_GetIntParam(c_param_name)
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

  
NESTGPU_SetIntParam = _nestgpu.NESTGPU_SetIntParam
NESTGPU_SetIntParam.argtypes = (c_char_p, ctypes.c_int)
NESTGPU_SetIntParam.restype = ctypes.c_int
def SetIntParam(param_name, val):
    "Set kernel int parameter value"

    c_param_name = ctypes.create_string_buffer(to_byte_str(param_name),
                                               len(param_name)+1)
    ret = NESTGPU_SetIntParam(c_param_name, ctypes.c_int(val))
    
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

def GetKernelStatus(var_key=None):
    "Get kernel status"
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetKernelStatus(var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        status_dict = {}
        name_list = GetFloatParamNames() + GetIntParamNames() + GetBoolParamNames()
        for param_name in name_list:
            val = GetKernelStatus(param_name)
            status_dict[param_name] = val
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
        if IsFloatParam(var_key):
            return GetFloatParam(var_key)        
        elif IsIntParam(var_key):
            return GetIntParam(var_key)
        elif IsBoolParam(var_key):
            return GetBoolParam(var_key)
        else:
            raise ValueError("Unknown parameter in GetKernelStatus", var_key)
    else:
        raise ValueError("Unknown key type in GetSynGroupStatus", type(var_key))

def SetKernelStatus(params, val=None):
    "Set kernel parameters using dictionaries"
    if ((type(params)==dict) & (val==None)):
        for param_name in params:
            SetKernelStatus(param_name, params[param_name])
    elif (type(params)==str):
        if IsFloatParam(params):
            return SetFloatParam(params, val)        
        elif IsIntParam(params):
            return SetIntParam(params, val)
        elif IsBoolParam(params):
            return SetBoolParam(params, val)
        else:
            raise ValueError("Unknown parameter in SetKernelStatus", params)
    else:
        raise ValueError("Wrong argument in SetKernelStatus")       
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())


NESTGPU_RemoteCreate = _nestgpu.NESTGPU_RemoteCreate
NESTGPU_RemoteCreate.argtypes = (ctypes.c_int, c_char_p, ctypes.c_int,
                                   ctypes.c_int)
NESTGPU_Create.restype = ctypes.c_int
def RemoteCreate(i_host, model_name, n_node=1, n_ports=1, status_dict=None):
    "Create a remote neuron group"
    if (type(status_dict)==dict):
        remote_node_group = RemoteCreate(i_host, model_name, n_node, n_ports)
        SetStatus(remote_node_group, status_dict)
        return remote_node_group
        
    elif status_dict!=None:
        raise ValueError("Wrong argument in RemoteCreate")
    
    c_model_name = ctypes.create_string_buffer(to_byte_str(model_name), len(model_name)+1)
    i_node = NESTGPU_RemoteCreate(ctypes.c_int(i_host), c_model_name, ctypes.c_int(n_node),
                                    ctypes.c_int(n_ports))
    node_seq = NodeSeq(i_node, n_node)
    ret = RemoteNodeSeq(i_host, node_seq)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret
