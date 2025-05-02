%{
/* C++ string header, for string ops below */
#include <string>
#include <variant>

/* Implementation of yyFlexScanner */ 
#include "verilog_scanner.hpp"
#undef  YY_DECL
#define YY_DECL int verilog::VerilogScanner::yylex(verilog::VerilogParser::semantic_type * const lval, verilog::VerilogParser::location_type *loc )

/* typedef to make the returns for the tokens shorter */
using token = verilog::VerilogParser::token;

/* define yyterminate as this instead of NULL */
#define yyterminate() return( token::END )

/* msvc2010 requires that we exclude this header file. */
#define YY_NO_UNISTD_H

/* update location on matching */
#define YY_USER_ACTION loc->step(); loc->columns(yyleng);
%}

/* Make the generated scanner run in debug mode. */
%option debug

/* Cause the default rule (that unmatched scanner input is echoed to stdout) to be suppressed. */
%option nodefault

/* Inform flex that a derived NAME is implemented as a subclass of yyFlexLexer, so flex will place your actions in the member function foo::yylex() instead of yyFlexLexer::yylex(). */
%option yyclass="verilog::VerilogScanner"

/* Set yywrap always returns 1. yywrap is called when reaching EOF */
%option noyywrap

/* Suppress the warning message yyunput is defined but not used */
%option nounput
%option never-interactive
%option c++

/* Define inclusive/exclusive states */
%x in_comment 
%x in_attribute

/* Predefined rules */
NEWLINE         "\n"|"\r\n"
SPACE           " "|"\t"|"\f"
COMMENT_BEGIN   "/*"
COMMENT_END     "*/" 
COMMENT_LINE  "//".*\n

ATTRIBUTE_BEGIN "(*"
ATTRIBUTE_END   "*)"

/* Name rule http://www.asic-world.com/verilog/syntax1.html */
NAME           [_a-zA-Z][$_a-zA-Z0-9]*{0,1023}
/* 
   Check the escape rule inside character class 
   https://www.regular-expressions.info/charclass.html
 */
ESCAPED_NAME   \\[\\^!"#$%&',()*+\-.a-zA-Z0-9/{|}~[\]_:;<=>?@]+[\t\f ]
INTEGER        [1-9][0-9]*|0
BINARY         [+-]?[0-9]*"'"[Bb][01_xXzZ]+
OCTAL          [+-]?[0-9]*"'"[Oo][0-7_xXzZ]+ 
DECIMAL        [+-]?[0-9]*"'"[Dd][0-9_xXzZ]+
HEX            [+-]?[0-9]*"'"[Hh][0-9a-fA-F_xXzZ]+ 
REAL           [-+]?([0-9]*\.[0-9]+|[0-9]+)
EXP            [-+]?[0-9]+"."?[0-9]*"E"[-+]?[0-9]+


/* Token rule section begins */
%%
%{ /** Code executed at the beginning of yylex **/
   yylval = lval;
%}

<*>{SPACE} { /* ignore any space */ }
{NEWLINE}  { // Update line numbers 
             loc->lines();
             // return( token::NEWLINE ); 
           }


{COMMENT_LINE}  { loc->lines(); }

{COMMENT_BEGIN} { BEGIN(in_comment); }
<in_comment><<EOF>> { 
                      BEGIN(INITIAL);
                      std::cerr << "Unclosed comment at line " << loc->end.line << " col " << loc->end.column << '\n';
                      yyterminate();
                    }
<in_comment>{NEWLINE} { loc->lines(); }
<in_comment>. { /* ignore characters in comment */ }
<in_comment>{COMMENT_END} { BEGIN(INITIAL); }


{ATTRIBUTE_BEGIN} { BEGIN(in_attribute); }
<in_attribute><<EOF>> { 
                      BEGIN(INITIAL);
                      std::cerr << "Unclosed attribute at line " << loc->end.line << " col " << loc->end.column << '\n';
                      yyterminate();
                    }
<in_attribute>{NEWLINE} { loc->lines(); }
<in_attribute>. { /* ignore characters in comment */ }
<in_attribute>{ATTRIBUTE_END} { BEGIN(INITIAL); }


":"|"."|"{"|"}"|"["|"]"|","|"*"|";"|"="|"-"|"+"|"|"|"("|")" {
  return yytext[0];
}

module    { return token::MODULE; }
endmodule { return token::ENDMODULE; }
input     { return token::INPUT; } 
output    { return token::OUTPUT; }
inout     { return token::INOUT; }
reg       { return token::REG; }
wire      { return token::WIRE; } 
wor       { return token::WOR; }
wand      { return token::WAND; }
tri       { return token::TRI; }
trior     { return token::TRIOR; }
triand    { return token::TRIAND; }
SUPPLY0   { return token::SUPPLY0; }
SUPPLY1   { return token::SUPPLY1; }
assign    { return token::ASSIGN; }


{ESCAPED_NAME}  { 
  yylval->build<std::string>(yytext);
  return token::ESCAPED_NAME;
}

{NAME} { 
  yylval->build<std::string>( yytext );
  return token::NAME;
}

{INTEGER} {
            yylval->build<verilog::Constant>(verilog::Constant(yytext, verilog::ConstantType::INTEGER)); 
            return token::INTEGER;
          }

{BINARY} {
           yylval->build<verilog::Constant>(verilog::Constant(yytext, verilog::ConstantType::BINARY));
           return token::BINARY;
         }

{OCTAL} {
          yylval->build<verilog::Constant>(verilog::Constant(yytext, verilog::ConstantType::OCTAL));
          return token::OCTAL;
        }

{DECIMAL} {
            yylval->build<verilog::Constant>(verilog::Constant(yytext, verilog::ConstantType::DECIMAL));
            return token::DECIMAL;
          }

{HEX} {
        yylval->build<verilog::Constant>(verilog::Constant(yytext, verilog::ConstantType::HEX));
        return token::HEX;
      }

{REAL} {
         verilog::Constant c(yytext, verilog::ConstantType::REAL);
         return token::REAL;
       }

{EXP} {
        verilog::Constant c(yytext, verilog::ConstantType::EXP);
        return token::EXP;
      }




. {  
    /* Last rule catches any unmatched character */
    std::cerr << "Failed to match : " << yytext << '\n';
    yyterminate();
  }
%%


