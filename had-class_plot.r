library(tidygraph)
library(ggraph)
library(tidyverse)
library(scales)
library(showtext)

script_dir <- dirname(sys.frame(1)$ofile)
if (length(script_dir) == 0 || script_dir == "") {
  script_dir <- getwd()
}
helvetica_font_path <- file.path(script_dir, "resources", "Helvetica.ttc")

showtext_auto()
font_add("Helvetica", helvetica_font_path)

plot_had_dendrogram <- function(
  had_file         = "resources/had.csv",

  root_label_size  = 12,  # HAD
  super_label_size = 6,   # superclass
  class_label_size = 3,   # class_name

  root_label_color  = "auto",
  super_label_color = "auto",
  class_label_color = "auto",


  root_node_color   = "auto",
  super_node_color  = "auto",
  class_node_color  = "auto",

  root_node_size   = 8,
  super_node_size  = 6,
  class_node_size  = 4,

  node_alpha       = 0,
  edge_alpha       = 0.3,

  edge_linewidth   = 0.5,
  label_distance   = 1.05,  # 标签距离节点的倍数
  plot_margin      = 20,     # 图边距（mm）

  out_file         = "had_dendrogram.png",
  width            = 12,
  height           = 12,
  dpi              = 600
) {

  had <- readr::read_csv(had_file, show_col_types = FALSE) %>%
    dplyr::select(-any_of(c("Unnamed: 0", "...1"))) %>%
    mutate(
      class_name        = as.character(class_name),
      superclass_level0 = as.character(superclass_level0)
    )

  root_node <- tibble(
    id          = 1L,
    node_name   = "HAD",
    node_branch = "root",
    leaf        = FALSE
  )

  class_counts <- had %>%
    count(superclass_level0, name = "n_classes")

  super_nodes <- had %>%
    distinct(superclass_level0) %>%
    arrange(superclass_level0) %>%
    left_join(class_counts, by = "superclass_level0") %>%
    mutate(
      id          = row_number() + 1L,   # 从 2 开始编号
      node_name   = superclass_level0,
      node_branch = superclass_level0,
      leaf        = FALSE
    )

  leaf_nodes <- had %>%
    arrange(superclass_level0, class_name) %>%
    mutate(
      id          = row_number() + 1L + nrow(super_nodes),
      node_name   = class_name,
      node_branch = superclass_level0,
      leaf        = TRUE
    )

  nodes <- bind_rows(root_node, super_nodes, leaf_nodes)

  if (is.list(super_label_color) && !is.null(names(super_label_color))) {
    map_color <- function(name) {
      if (name %in% names(super_label_color)) {
        return(super_label_color[[name]])
      } else {
        return(NA_character_)
      }
    }

    nodes <- nodes %>%
      mutate(
        super_label_color_mapped = case_when(
          node_branch == "root" ~ NA_character_,
          !leaf & node_branch != "root" ~ sapply(node_name, map_color),
          TRUE ~ NA_character_
        )
      )
    use_super_label_color_map <- TRUE
  } else {
    nodes <- nodes %>%
      mutate(super_label_color_mapped = NA_character_)
    use_super_label_color_map <- FALSE
  }

  if (is.list(class_label_color) && !is.null(names(class_label_color))) {
    map_color <- function(name) {
      if (name %in% names(class_label_color)) {
        return(class_label_color[[name]])
      } else {
        return(NA_character_)
      }
    }

    nodes <- nodes %>%
      mutate(
        class_label_color_mapped = case_when(
          leaf ~ sapply(node_name, map_color),
          TRUE ~ NA_character_
        )
      )
    use_class_label_color_map <- TRUE
  } else {
    nodes <- nodes %>%
      mutate(class_label_color_mapped = NA_character_)
    use_class_label_color_map <- FALSE
  }

  if (is.list(super_node_color) && !is.null(names(super_node_color))) {
    map_color <- function(name) {
      if (name %in% names(super_node_color)) {
        return(super_node_color[[name]])
      } else {
        return(NA_character_)
      }
    }

    nodes <- nodes %>%
      mutate(
        super_node_color_mapped = case_when(
          node_branch == "root" ~ NA_character_,
          !leaf & node_branch != "root" ~ sapply(node_name, map_color),
          TRUE ~ NA_character_
        )
      )
    use_super_node_color_map <- TRUE
  } else {
    nodes <- nodes %>%
      mutate(super_node_color_mapped = NA_character_)
    use_super_node_color_map <- FALSE
  }

  # ===== 处理 class 节点颜色映射 =====
  if (is.list(class_node_color) && !is.null(names(class_node_color))) {
    map_color <- function(name) {
      if (name %in% names(class_node_color)) {
        return(class_node_color[[name]])
      } else {
        return(NA_character_)
      }
    }

    nodes <- nodes %>%
      mutate(
        class_node_color_mapped = case_when(
          leaf ~ sapply(node_name, map_color),
          TRUE ~ NA_character_
        )
      )
    use_class_node_color_map <- TRUE
  } else {
    nodes <- nodes %>%
      mutate(class_node_color_mapped = NA_character_)
    use_class_node_color_map <- FALSE
  }

  # ===== 同步 super_node_color 到下游 class 节点 =====
  # 如果设置了 super_node_color，将其颜色应用到对应的下游 class 节点
  if (use_super_node_color_map) {
    # 创建颜色映射函数
    map_super_color <- function(branch) {
      if (branch %in% names(super_node_color)) {
        return(super_node_color[[branch]])
      } else {
        return(NA_character_)
      }
    }

    nodes <- nodes %>%
      mutate(
        class_node_color_inherited = if_else(
          leaf,
          sapply(node_branch, map_super_color),
          NA_character_
        )
      )
    use_class_node_color_inherited <- TRUE
  } else {
    nodes <- nodes %>%
      mutate(class_node_color_inherited = NA_character_)
    use_class_node_color_inherited <- FALSE
  }


  edges_root_to_super <- super_nodes %>%
    transmute(
      from = 1L,
      to   = id
    )

  edges_super_to_leaf <- leaf_nodes %>%
    left_join(
      super_nodes %>%
        select(superclass_level0, super_id = id),
      by = "superclass_level0"
    ) %>%
    transmute(
      from = super_id,
      to   = id
    )

  edges <- bind_rows(edges_root_to_super, edges_super_to_leaf)

  branches <- nodes %>%
    filter(node_branch != "root") %>%
    distinct(node_branch) %>%
    pull(node_branch)

  pal <- setNames(hue_pal()(length(branches)), branches)
  pal <- c("root" = "#f39c12", pal)  # 给 root 一个固定颜色

  if (root_node_color != "auto") {
    pal["root"] <- root_node_color
  }


  if (use_super_node_color_map) {
    for (super_name in names(super_node_color)) {
      if (super_name %in% names(pal)) {
        pal[super_name] <- super_node_color[[super_name]]
      }
    }
  }


  if (use_super_label_color_map) {
    custom_colors <- unlist(super_label_color)
    pal <- c(pal, custom_colors)
  }


  if (use_class_label_color_map) {
    custom_colors <- unlist(class_label_color)
    pal <- c(pal, custom_colors)
  }


  if (use_class_node_color_map) {
    custom_colors <- unlist(class_node_color)
    pal <- c(pal, custom_colors)
  }


  activity_graph <- tbl_graph(nodes = nodes, edges = edges, directed = TRUE)


  p <- ggraph(activity_graph, layout = "dendrogram", circular = TRUE) +
    # 边
    geom_edge_diagonal(
      aes(color = node1.node_branch),
      alpha     = edge_alpha,
      linewidth = edge_linewidth
    ) +

    geom_node_point(
      aes(filter = node_branch == "root", color = if (root_node_color == "auto") node_branch else NULL),
      size  = root_node_size,
      alpha = node_alpha,
      color = if (root_node_color != "auto") root_node_color else NULL
    ) +
    geom_node_point(
      aes(
        filter = !leaf & node_branch != "root" & (if (use_super_node_color_map) is.na(super_node_color_mapped) else TRUE),
        color = if (!use_super_node_color_map && super_node_color == "auto") node_branch else NULL,
        size = n_classes
      ),
      alpha = node_alpha,
      color = if (!use_super_node_color_map && super_node_color != "auto") super_node_color else NULL
    ) +
    {
      if (use_super_node_color_map) {
        lapply(names(super_node_color), function(super_name) {
          geom_node_point(
            aes(
              filter = !leaf & node_branch != "root" & node_name == super_name,
              size = n_classes
            ),
            alpha = node_alpha,
            color = super_node_color[[super_name]]
          )
        })
      }
    } +
    geom_node_point(
      aes(
        filter = leaf &
          (if (use_class_node_color_map) is.na(class_node_color_mapped) else TRUE) &
          (if (use_class_node_color_inherited) is.na(class_node_color_inherited) else TRUE),
        color = if (!use_class_node_color_map && class_node_color == "auto") node_branch else NULL
      ),
      size  = class_node_size,
      alpha = node_alpha,
      color = if (!use_class_node_color_map && class_node_color != "auto") class_node_color else NULL
    ) +
    {
      if (use_class_node_color_inherited) {
        lapply(names(super_node_color), function(super_name) {
          geom_node_point(
            aes(filter = leaf & node_branch == super_name &
                (if (use_class_node_color_map) is.na(class_node_color_mapped) else TRUE)),
            size  = class_node_size,
            alpha = node_alpha,
            color = super_node_color[[super_name]]
          )
        })
      }
    } +
    {
      if (use_class_node_color_map) {
        lapply(names(class_node_color), function(class_name) {
          geom_node_point(
            aes(filter = leaf & node_name == class_name),
            size  = class_node_size,
            alpha = node_alpha,
            color = class_node_color[[class_name]]
          )
        })
      }
    } +

    # 叶子：class_name（未使用自定义颜色、未继承颜色的）
    geom_node_text(
      aes(
        x      = x * label_distance,
        y      = y * label_distance,
        label  = node_name,
        angle  = node_angle(x, y),
        filter = leaf &
          (if (use_class_label_color_map) is.na(class_label_color_mapped) else TRUE) &
          (if (use_class_node_color_inherited) is.na(class_node_color_inherited) else TRUE),
        color  = if (!use_class_label_color_map && class_label_color == "auto") node_branch else NULL
      ),
      size  = class_label_size,
      hjust = "outward",
      color = if (!use_class_label_color_map && class_label_color != "auto") class_label_color else NULL
    ) +
    {
      if (use_class_node_color_inherited) {
        lapply(names(super_node_color), function(super_name) {
          geom_node_text(
            aes(
              x      = x * label_distance,
              y      = y * label_distance,
              label  = node_name,
              angle  = node_angle(x, y),
              filter = leaf & node_branch == super_name &
                (if (use_class_label_color_map) is.na(class_label_color_mapped) else TRUE)
            ),
            size  = class_label_size,
            hjust = "outward",
            color = super_node_color[[super_name]]
          )
        })
      }
    } +
    {
      if (use_class_label_color_map) {
        lapply(names(class_label_color), function(class_name) {
          geom_node_text(
            aes(
              x      = x * label_distance,
              y      = y * label_distance,
              label  = node_name,
              angle  = node_angle(x, y),
              filter = leaf & node_name == class_name
            ),
            size  = class_label_size,
            hjust = "outward",
            color = class_label_color[[class_name]]
          )
        })
      }
    } +

    geom_node_text(
      aes(
        label  = node_name,
        filter = node_branch == "root",
        color  = if (root_label_color == "auto") node_branch else NULL
      ),
      fontface = "bold",
      size     = root_label_size,
      color    = if (root_label_color != "auto") root_label_color else NULL
    ) +

    geom_node_text(
      aes(
        label  = node_name,
        filter = !leaf & node_branch != "root" & (if (use_super_label_color_map) is.na(super_label_color_mapped) else TRUE),
        color  = if (!use_super_label_color_map && super_label_color == "auto") node_branch else NULL
      ),
      fontface = "bold",
      size     = super_label_size,
      color    = if (!use_super_label_color_map && super_label_color != "auto") super_label_color else NULL
    ) +
    {
      if (use_super_label_color_map) {
        lapply(names(super_label_color), function(super_name) {
          geom_node_text(
            aes(
              label  = node_name,
              filter = !leaf & node_branch != "root" & node_name == super_name
            ),
            fontface = "bold",
            size     = super_label_size,
            color    = super_label_color[[super_name]]
          )
        })
      }
    } +

    scale_color_manual(values = pal) +
    scale_edge_color_manual(values = pal) +
    scale_size_continuous(range = c(super_node_size * 0.5, super_node_size * 2)) +  # 大小范围
    coord_fixed(clip = "off") +
    theme_void() +
    theme(
      legend.position = "none",
      plot.background = element_rect(fill = "transparent", color = NA),
      plot.margin     = margin(plot_margin, plot_margin, plot_margin, plot_margin, unit = "mm"),
      text            = element_text(family = "Helvetica")
    )

  print(p)

  ggsave(
    out_file,
    plot   = p,
    width  = width,
    height = height,
    dpi    = dpi,
    bg     = "transparent"
  )

  cat("图形已保存至:", out_file, "\n")
  cat("- 节点总数:", nrow(nodes), "\n")
  cat("- 大类数量:", nrow(super_nodes), "\n")
  cat("- 叶子节点:", nrow(leaf_nodes), "\n")

  invisible(p)
}

plot_had_dendrogram(
  root_label_size  = 15*3,
  super_label_size = 8*3,
  class_label_size = 3.5*3,

  root_label_color  = "#be74ec",
  root_node_color   = "#eccdff",
  super_label_color = list(
    "Sports" = "#0a8558",
    "Personal Care" = "#b17770",
    "Housework" = "#a1943c",
    "Socializing" = "#1c869c",
    "Eating" = "#5c912d"
  ),
  super_node_color = list(
    "Sports" = "#0ba869",
    "Personal Care" = "#df978d",
    "Housework" = "#cfc050",
    "Socializing" = "#21a5c0",
    "Eating" = "#73b539"
  ),
  root_node_size   = 50,
  super_node_size  = 20,
  class_node_size  = 4,
  node_alpha       = 0.5,
  edge_alpha       = 0.5,
  edge_linewidth   = 0.4,
  label_distance   = 1.05,
  plot_margin      = 50,
  out_file         = "/Users/zgh/Desktop/workingdir/HAD-test/had_class.png",
  dpi=250
)
