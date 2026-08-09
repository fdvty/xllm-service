include(cc_binary)
include(cc_library)
include(cc_test)

add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(runtime)
add_subdirectory(backend_http)
add_subdirectory(telemetry)
add_subdirectory(routing)
add_subdirectory(core)
add_subdirectory(request)
add_subdirectory(transport)
add_subdirectory(execution)
add_subdirectory(dispatcher)
add_subdirectory(rpc_service)
add_subdirectory(tokenizer)
add_subdirectory(chat_template)
add_subdirectory(http_service)
add_subdirectory(scheduler)

cc_binary(
  NAME
    xllm_master_serving
  HDRS
    master.h
  SRCS
    master.cpp
  DEPS
    :dispatcher
    :routing
    :xllm_http_service
    :xllm_rpc_service
    :runtime
    :transport
)

add_subdirectory(examples)
