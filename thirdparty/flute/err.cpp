#include <stdio.h>
#include <stdlib.h>

namespace Flute {

/**************************************************************************/
/*
  print error message and continue
*/

void  err_msg(
const char* msg
)
{
  fprintf(stderr, "%s\n", msg);
}

/**************************************************************************/
/*
  print error message and  exit
*/

void  err_exit(
const char* msg
)
{
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

}  // namespace Flute