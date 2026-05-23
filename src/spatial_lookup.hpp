#ifndef SPATIAL_LOOKUP_HPP_
#define SPATIAL_LOOKUP_HPP_

#include <algorithm>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/geometric.hpp>

class SpatialLookup {
public:
    struct Cell {
        size_t index;
        size_t key;
    };

public:
    SpatialLookup() = default;

    void Resize(size_t count) {
        m_spatial_lookup.resize(count);
        m_start_indices.resize(count);
    }

    size_t GetKeyFromHash(size_t hash) {
        return hash % m_spatial_lookup.size();
    }

    template <typename F>
    void ForEachNeighbor(float radius, glm::vec2 position, const std::vector<glm::vec2>& positions, F&& func) {
        const float sqr_radius = radius * radius;

        const glm::ivec2 center = PositionToCellCoord(position, radius);

        for (const glm::ivec2& offset : CELL_OFFSETS) {
            size_t key = GetKeyFromHash(HashCell(center + offset));
            size_t start_index = m_start_indices[key];

            for (size_t i = start_index; i < m_spatial_lookup.size(); ++i) {
                if (m_spatial_lookup[i].key != key) break;

                size_t particle_index = m_spatial_lookup[i].index;
                glm::vec2 offset = positions[particle_index] - position;
                float sqr_dst = glm::dot(offset, offset);

                if (sqr_dst < sqr_radius) {
                    float dst = glm::sqrt(sqr_dst);
                    std::forward<F>(func)(particle_index, offset, dst);
                }
            }
        }
    }

    void Sort() {
        std::sort(
            m_spatial_lookup.begin(),
            m_spatial_lookup.end(),
            [](const Cell& a, const Cell& b) {
                return a.key < b.key;
            }
        );
    }

    Cell GetCell(size_t index) const {
        return m_spatial_lookup[index];
    }

    size_t GetStartIndex(size_t index) const {
        return m_start_indices[index];
    }

    void SetCell(size_t index, Cell value) {
        m_spatial_lookup[index] = value;
    }

    void SetStartIndex(size_t index, size_t value) {
        m_start_indices[index] = value;
    }

    std::vector<Cell>& Cells() {
        return m_spatial_lookup;
    }

    const std::vector<Cell>& Cells() const {
        return m_spatial_lookup;
    }

    std::vector<size_t>& StartIndices() {
        return m_start_indices;
    }

    const std::vector<size_t>& StartIndices() const {
        return m_start_indices;
    }

    size_t GetSize() const {
        return m_spatial_lookup.size();
    }

public:
    static glm::ivec2 PositionToCellCoord(glm::vec2 position, float radius) {
        int x = position.x / radius;
        int y = position.y / radius;
        return {x, y};
    }

    static size_t HashCell(glm::ivec2 pos) {
        size_t a = pos.x * 15823;
        size_t b = pos.y * 9737333;
        return a + b;
    }

public:
    static glm::ivec2 CELL_OFFSETS[9];

private:
    std::vector<Cell> m_spatial_lookup;
    std::vector<size_t> m_start_indices;
};

#endif // SPATIAL_LOOKUP_HPP_