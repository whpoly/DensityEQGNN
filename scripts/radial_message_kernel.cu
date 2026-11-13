extern "C" __global__ void radial_messages(
    const long long* offsets, const long long* source,
    const float* distance, const float* direction, const float* envelope,
    const float* table, const float* scalar, const float* vector,
    const float* projection, float* out_s, float* out_v,
    int queries, int channels, int intervals, float cutoff) {
  int q = blockIdx.x;
  int c = threadIdx.x;
  if (q >= queries || c >= channels) return;
  float a = 0.0f, bx = 0.0f, by = 0.0f, bz = 0.0f;
  for (long long e = offsets[q]; e < offsets[q+1]; ++e) {
    float r = distance[e];
    if (r < 0.0f || r >= cutoff) continue;
    float u = r * (float(intervals) / cutoff);
    int lo = int(u);
    if (lo >= intervals) lo = intervals - 1;
    float t = u - float(lo);
    long long i = (long long)lo * (3*channels) + c;
    float fv = table[i] * (1.0f-t) + table[i+3*channels] * t;
    float fe = table[i+channels] * (1.0f-t) + table[i+4*channels] * t;
    float fs = table[i+2*channels] * (1.0f-t) + table[i+5*channels] * t;
    float w = envelope[e];
    fv *= w; fe *= w; fs *= w;
    long long atom = source[e];
    float ge = projection[atom*channels+c] * fe;
    a += scalar[atom*channels+c] * fs;
    bx += vector[(atom*3)*channels+c] * fv + ge * direction[e*3];
    by += vector[(atom*3+1)*channels+c] * fv + ge * direction[e*3+1];
    bz += vector[(atom*3+2)*channels+c] * fv + ge * direction[e*3+2];
  }
  out_s[q*channels+c] = a;
  out_v[(q*3)*channels+c] = bx;
  out_v[(q*3+1)*channels+c] = by;
  out_v[(q*3+2)*channels+c] = bz;
}
