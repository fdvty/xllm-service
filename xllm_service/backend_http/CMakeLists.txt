include(cc_library)
include(cc_test)

cc_library(
  NAME
    backend_http
  HDRS
    request_context.h
  SRCS
    request_context.cpp
  DEPS
    :routing
    gflags::gflags
)
target_link_libraries(backend_http PRIVATE brpc-static)

cc_test(
  NAME
    request_context_test
  SRCS
    request_context_test.cpp
  DEPS
    :backend_http
    GTest::gtest_main
)
if(TARGET request_context_test)
  target_link_libraries(request_context_test PRIVATE brpc-static)
endif()
