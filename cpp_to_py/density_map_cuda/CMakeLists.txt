  set(TARGET_NAME density_map_cuda)

add_pytorch_extension(${TARGET_NAME}
  ${TARGET_NAME}.cpp
  ${TARGET_NAME}_kernel.cu
  ${TARGET_NAME}_naive_kernel.cu)

install(TARGETS 
  ${TARGET_NAME}
  DESTINATION ${XPLACE_LIB_DIR})