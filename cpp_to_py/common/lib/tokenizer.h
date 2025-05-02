#pragma once
#include "common/common.h"
#include "common/lib/Helper.h"

namespace gt {

// string conversion
std::string to_lower(std::string);
std::string to_upper(std::string);

std::string remove_quote(std::string);
std::string unquoted(std::string);

// check string syntax
bool is_numeric(const std::string&);
bool is_array(const std::string&);
bool is_word(const std::string&);

template <typename I>
auto find_quote_pair(const I b, const I e) {
  
  if(auto l = std::find(b, e, '"'); l == e) {
    return std::make_pair(e, e);
  }
  else {
    if(auto r = std::find(std::next(l), e, '"'); r == e) {
      return std::make_pair(e, e);
    }
    else {
      return std::make_pair(l, r);
    }
  }
}

template <typename I>
auto find_brace_pair(const I b, const I e) {
  
  auto l = std::find(b, e, '{');
  auto r = l;

  int stack = 0;

  while(r != e) {
    if(*r == '{') ++stack;
    else if(*r == '}') --stack;
    if(stack == 0) break;
    ++r;
  }

  if(l == e || r == e) {
    return std::make_pair(e, e);
  }
  
  return std::make_pair(l, r);
}

template <typename I>
auto find_bracket_pair(const I b, const I e) {

  auto l = std::find(b, e, '[');
  auto r = l;

  int stack = 0;

  while(r != e) {
    if(*r == '[') ++stack;
    else if(*r == ']') --stack;
    if(stack == 0) break;
    ++r;
  }

  if(l == e || r == e) {
    return std::make_pair(e, e);
  }
  
  return std::make_pair(l, r);
}

template <typename I, typename C>
auto on_next_parentheses(const I b, const I e, C&& c) {

  auto l = std::find(b, e, "(");
  auto r = l;
  //std::find(l, e, ")");

  int stack = 0;

  while(r != e) {
    if(*r == "(") {
      ++stack;
    }
    else if(*r == ")") {
      --stack;
    }
    if(stack == 0) {
      break;
    }
    ++r;
  }
  
  if(l == e || r == e) return e;

  for(++l; l != r; ++l) {
    c(*l);
  }

  return r;
}

std::vector<std::string> tokenize(const std::filesystem::path&, std::string_view="", std::string_view="");
std::vector<std::string> split(const std::string&, std::string_view="");

};  // namespace gt
