#include "gpudp/lg/legalization_db.h"

namespace dp {

template <typename T>
struct FillerBlank {
    T xl;
    T yl;
    T xh;
    T yh;
    int bucket_list_level;

    FillerBlank() {}
    FillerBlank(T xl_, T yl_, T xh_, T yh_) : xl(xl_), yl(yl_), xh(xh_), yh(yh_) {}
    void intersect(const FillerBlank& rhs) {
        xl = std::max(xl, rhs.xl);
        xh = std::min(xh, rhs.xh);
        yl = std::max(yl, rhs.yl);
        yh = std::min(yh, rhs.yh);
    }
};

void fixCells2Bins(const LegalizationData& db,
                   const float* x,
                   const float* y,
                   const float* node_size_x,
                   const float* node_size_y,
                   float bin_size_x,
                   float bin_size_y,
                   float xl,
                   float yl,
                   float xh,
                   float yh,
                   int num_bins_x,
                   int num_bins_y,
                   int num_nodes,
                   int num_movable_nodes,
                   int num_conn_movable_nodes,
                   std::vector<std::vector<int>>& bin_cells) {
    // do not handle large macros
    // one cell cannot be distributed to one bin
    for (int i = 0; i < num_nodes; i += 1) {
        if (i < num_conn_movable_nodes || i >= num_movable_nodes) {
            int bin_id_x = (x[i] + node_size_x[i] / 2 - xl) / bin_size_x;
            int bin_id_y = (y[i] + node_size_y[i] / 2 - yl) / bin_size_y;

            bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x - 1);
            bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y - 1);

            int bin_id = bin_id_x * num_bins_y + bin_id_y;

            bin_cells[bin_id].push_back(i);
        }
    }
    // sort bin cells
    for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
        std::vector<int>& cells = bin_cells.at(i);
        std::sort(cells.begin(), cells.end(), [&](int node_id1, int node_id2) {
            float x1 = x[node_id1];
            float x2 = x[node_id2];
            return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
        });
    }
}

void reduceBlanks(const float* x,
                  const float* y,
                  const float* node_size_x,
                  const float* node_size_y,
                  const std::vector<std::vector<int>>& bin_cells,
                  float bin_size_x,
                  float bin_size_y,
                  float xl,
                  float yl,
                  float xh,
                  float yh,
                  float site_width,
                  float row_height,
                  int num_bins_x,
                  int num_bins_y,
                  std::vector<std::vector<FillerBlank<float>>>& bin_blanks) {
    for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
        int bin_id_x = i / num_bins_y;
        int bin_id_y = i - bin_id_x * num_bins_y;
        int bin_id = bin_id_x * num_bins_y + bin_id_y;

        float bin_xl = xl + bin_id_x * bin_size_x;
        float bin_xh = std::min(bin_xl + bin_size_x, xh);
        float bin_yl = yl + bin_id_y * bin_size_y;
        float bin_yh = std::min(bin_yl + bin_size_y, yh);

        FillerBlank<float> blank;
        blank.xl = floorDiv((bin_xl - xl), site_width) * site_width + xl;  // align blanks to sites
        blank.xh = floorDiv((bin_xh - xl), site_width) * site_width + xl;  // align blanks to sites
        blank.yl = bin_yl;
        blank.yh = bin_yl + row_height;

        bin_blanks.at(bin_id).push_back(blank);

        const std::vector<int>& cells = bin_cells.at(i);
        std::vector<FillerBlank<float>>& blanks = bin_blanks.at(bin_id);

        for (unsigned int ci = 0; ci < cells.size(); ++ci) {
            int node_id = cells.at(ci);
            float node_xl = x[node_id];
            float node_yl = y[node_id];
            float node_xh = node_xl + node_size_x[node_id];
            float node_yh = node_yl + node_size_y[node_id];

            if (blanks.empty()) {
                break;
            }
            FillerBlank<float>& blank = blanks.back();

            if (node_xh > blank.xl && node_xl < blank.xh)  // overlap
            {
                if (node_xl == blank.xl && node_xh == blank.xh)  // remove
                {
                    blanks.pop_back();
                }
                if (node_xl == blank.xl && node_xh < blank.xh)  // reduce
                {
                    blank.xl = node_xh;
                }
                if (node_xl > blank.xl && node_xh == blank.xh)  // reduce
                {
                    blank.xh = node_xl;
                }
                if (node_xl > blank.xl && node_xh < blank.xh)  // split
                {
                    FillerBlank<float> new_blank(node_xh, blank.yl, blank.xh, blank.yh);
                    blank.xh = node_xl;
                    blanks.push_back(new_blank);
                }
            }
        }
    }

    // // print blanks
    // for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
    //     const std::vector<FillerBlank<float>>& blanks = bin_blanks.at(i);
    //     for (unsigned int j = 0; j < blanks.size(); ++j) {
    //         const FillerBlank<float>& blank = blanks.at(j);
    //         logger.info(
    //             "bin %d: blank %d: xl = %g, xh = %g, yl = %g, yh = %g", i, j, blank.xl, blank.xh, blank.yl, blank.yh);
    //     }
    // }
}

void fillerLegalization(DPTorchRawDB& at_db) {
    LegalizationData db(at_db);

    int num_blanks_x = 1;
    int num_bins_x = 1;

    // bin dimension in y direction for blanks is different from that for cells
    int num_blanks_y = floorDiv((db.yh - db.yl), db.row_height);
    int num_bins_y = num_blanks_y;
    logger.info("%s num_blanks_y = %d", "Standard cell legalization", num_blanks_y);

    // adjust bin sizes
    float bin_size_x = (db.xh - db.xl) / static_cast<float>(num_bins_x);
    float bin_size_y = db.row_height;

    // allocate bin blanks
    std::vector<std::vector<FillerBlank<float>>> bin_blanks(num_blanks_x * num_blanks_y);
    std::vector<std::vector<FillerBlank<float>>> bin_blanks_copy(num_blanks_x * num_blanks_y);
    std::vector<std::vector<int>> bin_cells(num_bins_x * num_bins_y);

    // distribute cells to bins
    fixCells2Bins(db,
                  db.x,
                  db.y,
                  db.node_size_x,
                  db.node_size_y,
                  bin_size_x,
                  db.row_height,
                  db.xl,
                  db.yl,
                  db.xh,
                  db.yh,
                  num_bins_x,
                  num_bins_y,
                  db.num_nodes,
                  db.num_movable_nodes,
                  db.num_conn_movable_nodes,
                  bin_cells);

    // distribute blanks to bins
    reduceBlanks(db.x,
                 db.y,
                 db.node_size_x,
                 db.node_size_y,
                 bin_cells,
                 bin_size_x,
                 db.row_height,
                 db.xl,
                 db.yl,
                 db.xh,
                 db.yh,
                 db.site_width,
                 db.row_height,
                 num_bins_x,
                 num_bins_y,
                 bin_blanks);

    // sort all blanks and create blank bucket list
    int maxDegree = 0;
    robin_hood::unordered_map<int, vector<FillerBlank<float>>> blank_bucket_list;
    for (int i = 0; i < num_blanks_x * num_blanks_y; i += 1) {
        std::vector<FillerBlank<float>>& blanks = bin_blanks.at(i);
        for (unsigned int j = 0; j < blanks.size(); ++j) {
            FillerBlank<float>& blank = blanks.at(j);
            blank.bucket_list_level = roundDiv(blank.xh - blank.xl, db.site_width);
            maxDegree = std::max(maxDegree, blank.bucket_list_level);
            blank_bucket_list[blank.bucket_list_level].push_back(blank);
        }
    }
    logger.info("%s maxDegree = %d", "Blanks", maxDegree);

    // sorted filler cell id
    vector<int> fillers_to_blank(db.num_movable_nodes - db.num_conn_movable_nodes);
    for (int i = 0; i < db.num_movable_nodes - db.num_conn_movable_nodes; i += 1) {
        fillers_to_blank[i] = db.num_conn_movable_nodes + i;
    }
    std::sort(fillers_to_blank.begin(), fillers_to_blank.end(), [&](int node_id1, int node_id2) {
        float size_x1 = db.node_size_x[node_id1];
        float size_x2 = db.node_size_x[node_id2];
        return size_x1 < size_x2 || (size_x1 == size_x2 && node_id1 < node_id2);
    });
    // for (int i = 0; i < db.num_movable_nodes - db.num_conn_movable_nodes; i += 1) {
    //     logger.info("filler %d: size_x = %g", filler_id[i], db.node_size_x[filler_id[i]]);
    // }

    // put filler cells into blanks
    while (!fillers_to_blank.empty()) {
        int filler_id = fillers_to_blank.back();
        int filler_size_int = roundDiv(db.node_size_x[filler_id], db.site_width);
        FillerBlank<float>& max_blank = blank_bucket_list[maxDegree].back();
        assert(max_blank.bucket_list_level >= filler_size_int);
        // update filler position
        db.x[filler_id] = max_blank.xl;
        db.y[filler_id] = max_blank.yl;
        max_blank.xl += db.node_size_x[filler_id];
        max_blank.bucket_list_level -= filler_size_int;
        blank_bucket_list[max_blank.bucket_list_level].push_back(max_blank);
        blank_bucket_list[maxDegree].pop_back();
        while (blank_bucket_list[maxDegree].empty()) {
            maxDegree -= 1;
        }

        fillers_to_blank.pop_back();

        // logger.info("put filler %d into blank %d: blank_xl = %g, blank_xh = %g",
        //             filler_id,
        //             max_blank.bucket_list_level,
        //             max_blank.xl,
        //             max_blank.xh);

        // logger.info("num_remaining_fillers = %d, max degree = %d", fillers_to_blank.size(), maxDegree);
    }
}

}  // namespace dp