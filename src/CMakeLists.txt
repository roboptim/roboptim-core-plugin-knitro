
MACRO(KNITRO_PLUGIN NAME)
    ADD_LIBRARY(roboptim-core-plugin-${NAME} MODULE roboptim-core-plugin-${NAME}.cc ${HEADERS})
    PKG_CONFIG_USE_DEPENDENCY(roboptim-core-plugin-${NAME} roboptim-core)
    TARGET_LINK_LIBRARIES(roboptim-core-plugin-${NAME} knitro)
    TARGET_LINK_LIBRARIES(roboptim-core-plugin-${NAME} boost_filesystem)
    
    SET_TARGET_PROPERTIES(roboptim-core-plugin-${NAME} PROPERTIES
      PREFIX ""
      SOVERSION 3.2.0)
    INSTALL(TARGETS roboptim-core-plugin-${NAME}
      DESTINATION ${ROBOPTIM_CORE_PLUGINDIR})
ENDMACRO()

KNITRO_PLUGIN(knitro)
