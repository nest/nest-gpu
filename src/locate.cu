__device__ int locate(int val, int *data, int n)
{
  int i_left = 0;
  int i_right = n-1;
  int i = (i_left+i_right)/2;
  while(i_right-i_left>1) {
    if (data[i] > val) i_right = i;
    else if (data[i]<val) i_left = i;
    else break;
    i=(i_left+i_right)/2;
  }

  return i;
}
