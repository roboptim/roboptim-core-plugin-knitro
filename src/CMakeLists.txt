ADD_LIBRARY(roboptim-core-plugin-knitro MODULE roboptim-core-plugin-knitro.cc ${HEADERS})
PKG_CONFIG_USE_DEPENDENCY(roboptim-core-plugin-knitro roboptim-core)
SET_TARGET_PROPERTIES(roboptim-core-plugin-knitro PROPERTIES
  PREFIX ""
  SOVERSION 0.1.0)
INSTALL(TARGETS roboptim-core-plugin-knitro
  DESTINATION ${ROBOPTIM_CORE_PLUGINDIR})
TARGET_LINK_LIBRARIES(roboptim-core-plugin-knitro knitro)
