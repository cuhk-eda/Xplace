

#pragma once

namespace gt {

// #define BLOCK_SIZE 512
#define BLOCK_SIZE 512
#define BLOCK_NUMBER(n) (((n) + (BLOCK_SIZE)-1) / BLOCK_SIZE)
// el rf rf: e0 l1 r0 f1

#define NUM_ATTR 4

// using index_type = int64_t;
using index_type = int;


// Overloadded.
template <typename... Ts>
struct Functors : Ts... { 
  using Ts::operator()... ;
};

template <typename... Ts>
Functors(Ts...) -> Functors<Ts...>;

}