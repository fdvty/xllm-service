include(cc_library)
include(cc_test)

cc_library(
  NAME
    core
  HDRS
    snapshot_holder.h
    cluster_snapshot.h
    event.h
    inflight_table.h
    cluster_state_actor.h
  SRCS
    inflight_table.cpp
    cluster_state_actor.cpp
  DEPS
    :common
    absl::synchronization
    glog::glog
    nlohmann_json::nlohmann_json
)
target_link_libraries(core PRIVATE brpc-static)
add_dependencies(core brpc-static)

cc_test(
  NAME
    core_test
  SRCS
    inflight_table_test.cpp
    cluster_state_actor_test.cpp
    snapshot_holder_test.cpp
  DEPS
    :core
    :common
    absl::synchronization
    glog::glog
    GTest::gtest_main
)
if(TARGET core_test)
  target_link_libraries(core_test PRIVATE brpc-static)
endif()
