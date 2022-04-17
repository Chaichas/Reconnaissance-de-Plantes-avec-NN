#include <stdint.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <cmocka.h>

#include "main.cpp"

static void test_parse_args(void **state) {

  /* no arg, too many args */
  assert_int_equal(-1, parse_args(1, (char*[]){"prog_loop"}));
  assert_int_equal(-1, parse_args(3, (char*[]){"prog_loop", "0x1060", "0x1060"}));

  /* valid arg */
  assert_int_equal(1060, parse_args(2, (char*[]){"prog_loop", "1060"})); 
}

int main() {
  int result = 0;
  const struct CMUnitTest tests[] = {
  
      cmocka_unit_test(test_parse_args),
  };
  result |= cmocka_run_group_tests_name("main", tests, NULL, NULL);

  return result;
}
