#ifndef SCANNER_VERILOG_HPP_
#define SCANNER_VERILOG_HPP_

#if ! defined(yyFlexLexerOnce)
#include <FlexLexer.h>
#endif

#include "verilog_parser.tab.hh"
#include "location.hh"

namespace verilog {

class VerilogScanner : public yyFlexLexer{
  public:

    VerilogScanner(std::istream *in) : yyFlexLexer(in) {
    };
    virtual ~VerilogScanner() {};

    //get rid of override virtual function warning
    using FlexLexer::yylex;

    virtual
      int yylex( verilog::VerilogParser::semantic_type * const lval, 
          verilog::VerilogParser::location_type *location );
    // YY_DECL defined in mc_lexer.l
    // Method body created by flex in mc_lexer.yy.cc

  private:
    /* yyval ptr */
    verilog::VerilogParser::semantic_type *yylval = nullptr;
};

} /* end namespace MC */

#endif 
