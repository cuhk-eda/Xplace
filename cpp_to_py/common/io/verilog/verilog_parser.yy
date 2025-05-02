%skeleton "lalr1.cc"
%require  "3.0"
%debug 
%defines 
%define api.namespace {verilog}
%define parser_class_name {VerilogParser}

%define parse.error verbose

%code requires{
  #include "verilog_data.hpp"

  namespace verilog {
    class ParserVerilogInterface;
    class VerilogScanner;
  }

// The following definitions is missing when %locations isn't used
# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif 

}

%parse-param { VerilogScanner &scanner }
%parse-param { ParserVerilogInterface *driver }

%code {
  #include <iostream>
  #include <cstdlib>
  #include <fstream>
  #include <utility>
  #include <tuple>
  
   /* include for all driver functions */
  #include "verilog_driver.hpp"

#undef yylex
#define yylex scanner.yylex
}

%define api.value.type variant
%define parse.assert


%left '-' '+'
%left '*' '/'
%left UMINUS

%token              END    0     "end of file"
%token              NEWLINE
%token              UNDEFINED 

/* Valid name (Identifiers) */
%token<std::string> NAME 
%token<std::string> ESCAPED_NAME  

%token<verilog::Constant> INTEGER BINARY OCTAL DECIMAL HEX REAL EXP

/* Keyword tokens */
%token MODULE ENDMODULE INPUT OUTPUT INOUT REG WIRE WAND WOR TRI TRIOR TRIAND SUPPLY0 SUPPLY1 ASSIGN


/* Nonterminal Symbols */
%type<std::string> valid_name  

%type<std::pair<verilog::PortDirection, verilog::ConnectionType>> port_type 
%type<verilog::Port> port_declarations port_decl port_decl_statements

%type<verilog::NetType> net_type
%type<verilog::Net> net_decl_statements net_decl 

%type<verilog::Constant> constant
%type<verilog::Assignment> assignment 
%type<std::vector<std::variant<std::string, verilog::NetBit, verilog::NetRange>>> lhs lhs_concat lhs_exprs lhs_expr
%type<std::vector<verilog::NetConcat>> rhs rhs_concat rhs_exprs rhs_expr 

%type<verilog::Instance> instance  
%type<std::pair<std::vector<std::variant<std::string, NetBit, NetRange>>, std::vector<std::vector<verilog::NetConcat>>>> inst_pins nets_by_name 

%type<std::vector<std::vector<verilog::NetConcat>>> nets_by_position

%type<std::pair<std::variant<std::string, NetBit, NetRange>, std::vector<verilog::NetConcat>>> net_by_name 

%locations 
%start design

%%

valid_name
  : NAME { $$ = $1; }
  | ESCAPED_NAME { $$ = $1; }
  ;

design 
  : modules;

modules
  :
  | modules module
  ;

module
  : MODULE valid_name ';' 
    { 
      driver->add_module(std::move($2));
    }
    statements ENDMODULE  
  | MODULE valid_name '(' ')' ';'
    {
      driver->add_module(std::move($2));
    }
    statements ENDMODULE
  | MODULE valid_name '(' port_names ')' ';' 
    {
      driver->add_module(std::move($2));
    }
    statements ENDMODULE
  | MODULE valid_name '(' 
    { 
      driver->add_module(std::move($2)); 
    } 
    port_declarations ')' 
    { 
      driver->add_port(std::move($5)); 
    }
    ';' statements ENDMODULE 
  ;

// port names are ignored as they will be parsed later in declaration
port_names 
  : valid_name { }
  | port_names ',' valid_name  { }
  ; 


port_type 
  : INPUT      { $$ = std::make_pair(verilog::PortDirection::INPUT, verilog::ConnectionType::NONE); }
  | INPUT WIRE { $$ = std::make_pair(verilog::PortDirection::INPUT, verilog::ConnectionType::WIRE); }
  | OUTPUT     { $$ = std::make_pair(verilog::PortDirection::OUTPUT,verilog::ConnectionType::NONE); }
  | OUTPUT REG { $$ = std::make_pair(verilog::PortDirection::OUTPUT,verilog::ConnectionType::REG);  }
  | INOUT      { $$ = std::make_pair(verilog::PortDirection::INOUT, verilog::ConnectionType::NONE); }
  | INOUT WIRE { $$ = std::make_pair(verilog::PortDirection::INOUT, verilog::ConnectionType::WIRE); }
  | INOUT REG  { $$ = std::make_pair(verilog::PortDirection::INOUT, verilog::ConnectionType::REG);  }
  ;

// e.g. "input a, b, output c, d" is allowed in port declarations
port_declarations
  : port_decl 
    {
      $$ = $1;
    }
  | port_declarations ',' port_decl  
    {
      driver->add_port(std::move($1));
      $$ = $3;
    }
  | port_declarations ',' valid_name 
    {
      $1.names.emplace_back(std::move($3));    
      $$ = $1;
    }
  ;

port_decl 
  : port_type valid_name 
    {
      $$.dir  = std::get<0>($1);
      $$.type = std::get<1>($1);
      $$.names.emplace_back(std::move($2)); 
    }
  | port_type '[' INTEGER ':' INTEGER ']' valid_name  
    {
      $$.dir  = std::get<0>($1);
      $$.type = std::get<1>($1);
      $$.beg = std::stoi($3.value);
      $$.end = std::stoi($5.value);
      $$.names.push_back(std::move($7)); 
    }
  ;

statements 
  : // empty
  | statements statement
  | statements statement_assign
  ; 

statement
  : declaration
  | instance
  ;


declaration 
  : port_decl_statements ';' { driver->add_port(std::move($1)); } 
  | net_decl_statements  ';' { driver->add_net(std::move($1)); }
  ;

// e.g. "input a, b, output c, d" is not allowed in port declaration statements 
port_decl_statements
  : port_decl 
    {
      $$ = $1;
    }
  | port_decl_statements ',' valid_name 
    {
      $1.names.emplace_back(std::move($3));    
      $$ = $1;
    }
  ;


net_type 
  :  WIRE    { $$ = verilog::NetType::WIRE;    }
  |  WAND    { $$ = verilog::NetType::WAND;    }
  |  WOR     { $$ = verilog::NetType::WOR;     }
  |  TRI     { $$ = verilog::NetType::TRI;     }
  |  TRIOR   { $$ = verilog::NetType::TRIOR;   }
  |  TRIAND  { $$ = verilog::NetType::TRIAND;  }
  |  SUPPLY0 { $$ = verilog::NetType::SUPPLY0; }
  |  SUPPLY1 { $$ = verilog::NetType::SUPPLY1; }
  ;

net_decl_statements
  : net_decl 
    {
      $$ = $1;
    }
  | net_decl_statements ',' valid_name 
    {
      $1.names.push_back(std::move($3));
      $$ = $1;
    }
  ;

net_decl 
  : net_type valid_name 
    {
      $$.type = $1;
      $$.names.push_back(std::move($2)); 
    }
  | net_type '[' INTEGER ':' INTEGER ']' valid_name  
    {
      $$.type = $1;
      $$.beg = std::stoi($3.value);
      $$.end = std::stoi($5.value);
      $$.names.push_back(std::move($7)); 
    }
  ;


statement_assign
  : ASSIGN assignments ';'

assignments
  : assignment 
  | assignments ',' assignment 
  ;

assignment
  : lhs '=' rhs { $$.lhs = $1; $$.rhs = $3; driver->add_assignment(std::move($$)); }
  ;


// Should try to merge lhs & rhs definition
lhs
  : valid_name { $$.push_back(std::move($1)); }
  | valid_name '[' INTEGER ']' 
    { $$.emplace_back(verilog::NetBit(std::move($1), std::stoi($3.value))); }
  | valid_name '[' INTEGER ':' INTEGER ']' 
    { $$.emplace_back(verilog::NetRange(std::move($1), std::stoi($3.value), std::stoi($5.value))); }
  | lhs_concat { $$ = $1; }
  ;
 
lhs_concat
  : '{' lhs_exprs '}' { std::move($2.begin(), $2.end(), std::back_inserter($$)); }
  ; 

lhs_exprs 
  : lhs_expr { std::move($1.begin(), $1.end(), std::back_inserter($$)); }
  | lhs_exprs ',' lhs_expr 
    { 
      std::move($1.begin(), $1.end(), std::back_inserter($$));
      std::move($3.begin(), $3.end(), std::back_inserter($$));
    } 
  ;

lhs_expr 
  : valid_name { $$.push_back(std::move($1)); }
  | valid_name '[' INTEGER ']' 
    { $$.emplace_back(verilog::NetBit(std::move($1), std::stoi($3.value))); }
  | valid_name '[' INTEGER ':' INTEGER ']' 
    { $$.emplace_back(verilog::NetRange(std::move($1), std::stoi($3.value), std::stoi($5.value))); }
  | lhs_concat 
    { std::move($1.begin(), $1.end(), std::back_inserter($$)); }
  ;



constant
  : INTEGER  { $$=$1; }
  | BINARY { $$=$1; }
  | OCTAL { $$=$1; } 
  | DECIMAL { $$=$1; }
  | HEX { $$=$1; }
  | REAL { $$=$1; }
  | EXP  { $$=$1; }
  ;

rhs
  : valid_name { $$.emplace_back($1); }
  | valid_name '[' INTEGER ']' 
    { $$.emplace_back(verilog::NetBit(std::move($1), std::stoi($3.value))); }
  | valid_name '[' INTEGER ':' INTEGER ']' 
    { $$.emplace_back(verilog::NetRange(std::move($1), std::stoi($3.value), std::stoi($5.value))); } 
  | constant { $$.push_back(std::move($1)); }
  | rhs_concat { $$ = $1; }
  ;
 
rhs_concat
  : '{' rhs_exprs '}' { std::move($2.begin(), $2.end(), std::back_inserter($$)); }
  ; 

rhs_exprs 
  : rhs_expr { std::move($1.begin(), $1.end(), std::back_inserter($$)); }
  | rhs_exprs ',' rhs_expr 
    { 
      std::move($1.begin(), $1.end(), std::back_inserter($$));
      std::move($3.begin(), $3.end(), std::back_inserter($$));
    } 
  ;

rhs_expr 
  : valid_name { $$.push_back(std::move($1)); }
  | valid_name '[' INTEGER ']' 
    { $$.emplace_back(verilog::NetBit(std::move($1), std::stoi($3.value))); }
  | valid_name '[' INTEGER ':' INTEGER ']' 
    { $$.emplace_back(verilog::NetRange(std::move($1), std::stoi($3.value), std::stoi($5.value))); } 
  | constant { $$.push_back(std::move($1)); }
  | rhs_concat 
    { std::move($1.begin(), $1.end(), std::back_inserter($$)); }
  ;





instance 
  : valid_name valid_name '(' inst_pins ')' ';'
    { 
      std::swap($$.module_name, $1); 
      std::swap($$.inst_name, $2); 
      std::swap($$.pin_names, std::get<0>($4));  
      std::swap($$.net_names, std::get<1>($4));  
      driver->add_instance(std::move($$));
    }
  | valid_name parameters valid_name '(' inst_pins ')' ';'
    { 
      std::swap($$.module_name, $1); 
      std::swap($$.inst_name, $3); 
      std::swap($$.pin_names, std::get<0>($5));  
      std::swap($$.net_names, std::get<1>($5));  
      driver->add_instance(std::move($$));
    }
  ;

inst_pins 
  : { } // empty
  | nets_by_position { std::swap(std::get<1>($$), $1); }
  | nets_by_name 
    {
      std::swap(std::get<0>($$), std::get<0>($1));
      std::swap(std::get<1>($$), std::get<1>($1));
    }
  ;

nets_by_position
  : rhs { $$.emplace_back(std::move($1)); }
  | nets_by_position ',' rhs
    { 
      std::move($1.begin(), $1.end(), std::back_inserter($$));   
      $$.push_back(std::move($3));
    }
  ;
  

nets_by_name 
  : net_by_name 
    { 
      std::get<0>($$).push_back(std::move(std::get<0>($1))); 
      std::get<1>($$).push_back(std::move(std::get<1>($1))); 
    }
  | nets_by_name ',' net_by_name 
    { 
      auto &pin_names = std::get<0>($1);
      auto &net_names = std::get<1>($1);
      std::move(pin_names.begin(), pin_names.end(), std::back_inserter(std::get<0>($$)));
      std::move(net_names.begin(), net_names.end(), std::back_inserter(std::get<1>($$)));

      std::get<0>($$).push_back(std::move(std::get<0>($3))); 
      std::get<1>($$).push_back(std::move(std::get<1>($3))); 
    }
  ;


net_by_name
  : '.' valid_name '(' ')' 
    { std::get<0>($$) = $2; } 
  | '.' valid_name '(' valid_name ')' 
    { 
       std::get<0>($$) = $2; 
       std::get<1>($$).push_back(std::move($4)); 
    } 
  | '.' valid_name '(' valid_name '[' INTEGER ']' ')' 
    { 
      std::get<0>($$) = $2; 
      std::get<1>($$).emplace_back(verilog::NetBit(std::move($4), std::stoi($6.value))); 
    } 
  // The previous two rules are also in rhs. But I don't want to create special rule just for this case 
  | '.' valid_name '(' rhs ')' 
    { 
      std::get<0>($$) = $2; 
      std::get<1>($$) = $4; 
    }
  // Bus port bit 
  | '.' valid_name '[' INTEGER ']' '(' ')'
    {
      std::get<0>($$) = verilog::NetBit(std::move($2), std::stoi($4.value)); 
    }
  | '.' valid_name '[' INTEGER ']' '(' rhs ')'
    {
      std::get<0>($$) = verilog::NetBit(std::move($2), std::stoi($4.value)); 
      std::get<1>($$) = $7; 
    }
  // Bus port part 
  | '.' valid_name '[' INTEGER ':' INTEGER ']' '(' ')'
    {
      std::get<0>($$) = verilog::NetRange(std::move($2), std::stoi($4.value), std::stoi($6.value)); 
    }
  | '.' valid_name '[' INTEGER ':' INTEGER ']' '(' rhs ')'
    {
      std::get<0>($$) = verilog::NetRange(std::move($2), std::stoi($4.value), std::stoi($6.value)); 
      std::get<1>($$) = $9; 
    }
  ;


// parameters are ignored for now
parameters 
  : '#' '(' param_exprs ')'
  ;

param_exprs
  : param_expr 
  | param_exprs ',' param_expr
  ;

param_expr 
  : valid_name 
  | '`' valid_name 
  | constant 
  | '-' param_expr %prec UMINUS 
  | param_expr '+' param_expr
  | param_expr '-' param_expr
  | param_expr '*' param_expr
  | param_expr '/' param_expr
  | '(' param_expr ')'
  ;


%%

void verilog::VerilogParser::error(const location_type &l, const std::string &err_message) {
  std::cerr << "Parser error: " << err_message  << '\n'
            << "  begin at line " << l.begin.line <<  " col " << l.begin.column  << '\n' 
            << "  end   at line " << l.end.line <<  " col " << l.end.column << "\n";
  std::abort();
}


