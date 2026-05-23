#include "spatial_lookup.hpp"

glm::ivec2 SpatialLookup::CELL_OFFSETS[] = { 
    glm::ivec2(0, 0),
    glm::ivec2(-1, 0),
    glm::ivec2(-1, -1),
    glm::ivec2(0, -1),
    glm::ivec2(1, -1),
    glm::ivec2(1, 0),
    glm::ivec2(1, 1),
    glm::ivec2(0, 1),
    glm::ivec2(-1, 1),
};