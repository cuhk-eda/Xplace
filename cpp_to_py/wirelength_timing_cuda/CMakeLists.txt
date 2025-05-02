set(TARGET_NAME wirelength_timing_cuda)

add_pytorch_extension(${TARGET_NAME}
  ${TARGET_NAME}.cpp
  ${TARGET_NAME}_kernel.cu)

install(TARGETS 
  ${TARGET_NAME}
  DESTINATION ${XPLACE_LIB_DIR})