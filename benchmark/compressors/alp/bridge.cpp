// ABI and packet serialization only; encoding/decoding calls the pinned upstream ALP library.
#include "alp.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>
using Bytes=std::vector<uint8_t>;
constexpr size_t N=alp::config::VECTOR_SIZE;
template<class T> void put(Bytes& b,T v){auto p=reinterpret_cast<const uint8_t*>(&v);b.insert(b.end(),p,p+sizeof(v));}
void append(Bytes& b,const void* p,size_t n){auto x=static_cast<const uint8_t*>(p);b.insert(b.end(),x,x+n);}
struct Read {const uint8_t* p;size_t n,pos=0;void bytes(void* out,size_t size){if(size>n-pos)throw std::runtime_error("truncated");memcpy(out,p+pos,size);pos+=size;}template<class T>T get(){T x;bytes(&x,sizeof(x));return x;}};
template<class T> void decode_vector(uint8_t mode,const Bytes& b,std::array<T,N>& out){
 using S=typename alp::inner_t<T>::st;using U=typename alp::inner_t<T>::ut;
 Read r{b.data(),b.size()};std::array<uint16_t,N> positions{},ex16{},left{},leftpack{};std::array<T,N> exceptions{};std::array<S,N> packed{},digits{};std::array<U,N> right{},rightpack{};
 uint16_t ec=0;
 if(mode==1){
  auto fac=r.get<uint8_t>(),exp=r.get<uint8_t>(),bw=r.get<uint8_t>();S base=r.get<S>();ec=r.get<uint16_t>();
  if(bw>sizeof(T)*8||ec>N||fac>alp::Constants<T>::MAX_EXPONENT||exp>alp::Constants<T>::MAX_EXPONENT)throw std::runtime_error("ALP header");
  r.bytes(exceptions.data(),ec*sizeof(T));r.bytes(positions.data(),ec*2);r.bytes(packed.data(),N*bw/8);
  for(size_t i=0;i<ec;i++)if(positions[i]>=N)throw std::runtime_error("ALP exception position");
  unffor::unffor(packed.data(),digits.data(),bw,&base);
  alp::decoder<T>::decode(digits.data(),fac,exp,out.data());alp::decoder<T>::patch_exceptions(out.data(),exceptions.data(),positions.data(),&ec);
 }else if(mode==2){
  alp::state<T> st;st.right_bit_width=r.get<uint8_t>();st.left_bit_width=r.get<uint8_t>();st.actual_dictionary_size=r.get<uint8_t>();ec=r.get<uint16_t>();
  if(st.right_bit_width>=sizeof(T)*8||st.left_bit_width>16||st.actual_dictionary_size>alp::config::MAX_RD_DICTIONARY_SIZE||ec>N)throw std::runtime_error("ALP RD header");
  r.bytes(st.left_parts_dict,st.actual_dictionary_size*2);r.bytes(ex16.data(),ec*2);r.bytes(positions.data(),ec*2);
  r.bytes(rightpack.data(),N*st.right_bit_width/8);r.bytes(leftpack.data(),N*st.left_bit_width/8);
  unffor::unffor(rightpack.data(),right.data(),st.right_bit_width,&st.right_for_base);unffor::unffor(leftpack.data(),left.data(),st.left_bit_width,&st.left_for_base);
  for(size_t i=0;i<N;i++)if(left[i]>=alp::config::MAX_RD_DICTIONARY_SIZE)throw std::runtime_error("ALP RD index");
  for(size_t i=0;i<ec;i++)if(positions[i]>=N)throw std::runtime_error("ALP RD exception position");
  alp::rd_encoder<T>::decode(out.data(),right.data(),left.data(),ex16.data(),positions.data(),&ec,st);
 }else throw std::runtime_error("ALP mode");
 if(r.pos!=r.n)throw std::runtime_error("ALP trailing data");
}
template<class T> Bytes encode(const uint8_t* raw,size_t size){
 if(size%sizeof(T))throw std::runtime_error("alignment");Bytes result;put<uint64_t>(result,size);
 for(size_t offset=0;offset<size;offset+=N*sizeof(T)){
  size_t used=std::min(size-offset,N*sizeof(T));std::array<T,N> in{},sample{},exceptions{},decoded{};memcpy(in.data(),raw+offset,used);
  using S=typename alp::inner_t<T>::st;using U=typename alp::inner_t<T>::ut;
  std::array<S,N> digits{},packed{};std::array<U,N> right{},rightpack{};std::array<uint16_t,N> positions{},ex16{},left{},leftpack{};
  uint16_t ec=0;alp::state<T> st;alp::encoder<T>::init(in.data(),0,N,sample.data(),st);Bytes b;uint8_t mode;
  if(st.scheme==alp::Scheme::ALP_RD){
   mode=2;alp::rd_encoder<T>::init(in.data(),0,N,sample.data(),st);alp::rd_encoder<T>::encode(in.data(),ex16.data(),positions.data(),&ec,right.data(),left.data(),st);
   ffor::ffor(right.data(),rightpack.data(),st.right_bit_width,&st.right_for_base);ffor::ffor(left.data(),leftpack.data(),st.left_bit_width,&st.left_for_base);
   put<uint8_t>(b,st.right_bit_width);put<uint8_t>(b,st.left_bit_width);put<uint8_t>(b,st.actual_dictionary_size);put<uint16_t>(b,ec);
   append(b,st.left_parts_dict,st.actual_dictionary_size*2);append(b,ex16.data(),ec*2);append(b,positions.data(),ec*2);append(b,rightpack.data(),N*st.right_bit_width/8);append(b,leftpack.data(),N*st.left_bit_width/8);
  }else{
   mode=1;alp::encoder<T>::encode(in.data(),exceptions.data(),positions.data(),&ec,digits.data(),st);uint8_t bw;S base;
   alp::encoder<T>::analyze_ffor(digits.data(),bw,&base);ffor::ffor(digits.data(),packed.data(),bw,&base);
   put<uint8_t>(b,st.fac);put<uint8_t>(b,st.exp);put<uint8_t>(b,bw);put<S>(b,base);put<uint16_t>(b,ec);append(b,exceptions.data(),ec*sizeof(T));append(b,positions.data(),ec*2);append(b,packed.data(),N*bw/8);
  }
  decode_vector<T>(mode,b,decoded);
  if(b.size()>=used||memcmp(in.data(),decoded.data(),used)!=0){mode=0;b.assign(raw+offset,raw+offset+used);}
  put<uint8_t>(result,mode);put<uint32_t>(result,b.size());append(result,b.data(),b.size());
 }
 return result;
}
template<class T> Bytes decode(const uint8_t* data,size_t size,size_t expected){
 Read r{data,size};if(r.get<uint64_t>()!=expected||expected%sizeof(T))throw std::runtime_error("ALP output size");Bytes out;
 while(out.size()<expected){uint8_t mode=r.get<uint8_t>();uint32_t n=r.get<uint32_t>();if(n>1024*1024)throw std::runtime_error("ALP frame size");Bytes b(n);r.bytes(b.data(),n);size_t used=std::min(expected-out.size(),N*sizeof(T));
  if(mode==0){if(n!=used)throw std::runtime_error("ALP raw size");append(out,b.data(),n);}else{std::array<T,N> values{};decode_vector<T>(mode,b,values);append(out,values.data(),used);}}
 if(r.pos!=r.n)throw std::runtime_error("ALP packet trailing data");return out;
}
extern "C" int64_t kv_alp_encode(const uint8_t* src,size_t n,int width,uint8_t* dst,size_t cap){try{Bytes b=width==4?encode<float>(src,n):encode<double>(src,n);if(b.size()>cap)return -2;memcpy(dst,b.data(),b.size());return b.size();}catch(...){return -1;}}
extern "C" int64_t kv_alp_decode(const uint8_t* src,size_t n,int width,uint8_t* dst,size_t expected){try{Bytes b=width==4?decode<float>(src,n,expected):decode<double>(src,n,expected);memcpy(dst,b.data(),b.size());return b.size();}catch(...){return -1;}}
