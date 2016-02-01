# Copyright 2014, Thomas Moulard, CNRS JRL UMI3218
# Copyright 2016, Benjamin Chrétien, CNRS-AIST JRL
#
# This file is part of roboptim-core.
# roboptim-core is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# roboptim-core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Lesser Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with roboptim-core.  If not, see <http://www.gnu.org/licenses/>.

MACRO(KNITRO_PLUGIN NAME)
    ADD_LIBRARY(roboptim-core-plugin-${NAME} MODULE ${NAME}.cc ${HEADERS})
    PKG_CONFIG_USE_DEPENDENCY(roboptim-core-plugin-${NAME} roboptim-core)
    TARGET_LINK_LIBRARIES(roboptim-core-plugin-${NAME} knitro)
    TARGET_LINK_LIBRARIES(roboptim-core-plugin-${NAME} boost_filesystem boost_thread)

    SET_TARGET_PROPERTIES(roboptim-core-plugin-${NAME} PROPERTIES
      PREFIX ""
      SOVERSION 3.2.0)
    INSTALL(TARGETS roboptim-core-plugin-${NAME}
      DESTINATION ${ROBOPTIM_CORE_PLUGINDIR})
ENDMACRO()

KNITRO_PLUGIN(knitro)
KNITRO_PLUGIN(knitro-sparse)
