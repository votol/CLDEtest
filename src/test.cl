__kernel void test(__global double* a_vec, __global double* b_vec) {

   int gid = get_global_id(0);
   
   a_vec[gid] += 0.01 * b_vec[gid];
}
