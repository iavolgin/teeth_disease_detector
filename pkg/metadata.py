import numpy as np
from scipy.interpolate import UnivariateSpline

def get_centers(bboxes):
    """
    Вычисляет центры bounding box'ов.
    :param bboxes: Список bbox в формате [x1, y1, x2, y2]
    :return: Массив центров shape (N, 2)
    """
    bboxes = np.array(bboxes)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.column_stack((cx, cy))

def get_global_bbox(bboxes):
    """
    Вычисляет общий bbox по всем зубам.
    :param bboxes: Список bbox
    :return: [min_x, min_y, max_x, max_y]
    """
    bboxes = np.array(bboxes)
    min_x = np.min(bboxes[:, 0])
    min_y = np.min(bboxes[:, 1])
    max_x = np.max(bboxes[:, 2])
    max_y = np.max(bboxes[:, 3])
    return np.array([min_x, min_y, max_x, max_y])

def fit_jaw_spline(centers, window_size=12, compression_factor=0.3):
    """
    Строит плавную кривую по центрам зубов для разделения на верхнюю и нижнюю челюсть.
    Сначала применяется скользящее среднее для сглаживания точек, затем строится сплайн.
    
    :param centers: Массив центров (N, 2)
    :param window_size: Размер окна для скользящего среднего (количество соседних точек)
    :param compression_factor: Коэффициент сжатия по Y (0.0 - 1.0)
    :return: Объект сплайна
    """
    sort_idx = np.argsort(centers[:, 0])
    x_sorted = centers[sort_idx, 0]
    y_sorted = centers[sort_idx, 1]

    _, unique_idx = np.unique(x_sorted, return_index=True)
    x_unique = x_sorted[unique_idx]
    y_unique = y_sorted[unique_idx]

    n_points = len(x_unique)
    
    if n_points < 4:
        mean_y = np.mean(y_unique)
        class ConstantSpline:
            def __call__(self, x):
                return np.full_like(x, mean_y)
        return ConstantSpline()

    y_smoothed = np.zeros_like(y_unique)
    
    for i in range(n_points):
        left = max(0, i - window_size // 2)
        right = min(n_points, i + window_size // 2 + 1)
        
        y_smoothed[i] = np.mean(y_unique[left:right])
    
    mean_y = np.mean(y_smoothed)
    deviations = y_smoothed - mean_y
    y_compressed = mean_y + deviations * compression_factor
    
    spline = UnivariateSpline(x_unique, y_compressed, s=n_points * 0.5)
    
    return spline

def compute_metadata(bboxes, image_width):
    """
    Основная функция вычисления метаданных для каждого bbox.
    
    :param bboxes: Список списков [x1, y1, x2, y2]
    :param image_width: Ширина исходного изображения (для определения центра изображения)
    :return: Список словарей с признаками
    """
    if not bboxes:
        return []

    bboxes_np = np.array(bboxes)
    centers = get_centers(bboxes)
    global_bbox = get_global_bbox(bboxes)
    
    min_x, min_y, max_x, max_y = global_bbox
    global_width = max_x - min_x
    global_height = max_y - min_y
    
    if global_width == 0: global_width = 1
    if global_height == 0: global_height = 1

    img_center_x = image_width / 2
    bbox_center_x = (min_x + max_x) / 2
    x_split_threshold = (img_center_x + bbox_center_x) / 2

    spline = fit_jaw_spline(centers)

    results = []
    n_teeth = len(bboxes)

    for i in range(n_teeth):
        x1, y1, x2, y2 = bboxes[i]
        cx, cy = centers[i]
        
        is_upper = cy < spline(cx)
        is_left = cx < x_split_threshold
        
        if is_upper:
            quadrant = 1 if is_left else 2
        else:
            quadrant = 4 if is_left else 3

        x_norm = (cx - min_x) / global_width
        y_norm = (cy - min_y) / global_height

        width = x2 - x1
        height = y2 - y1
        if width == 0: width = 1e-6
        aspect_ratio = height / width

        neighbours_right = int(np.sum(centers[:, 0] > cx))

        neighbours_top = int(np.sum(centers[:, 1] < cy))

        results.append({
            'quadrant': quadrant,
            'x_norm': x_norm,
            'y_norm': y_norm,
            'aspect_ratio': aspect_ratio,
            'neighbours_right': neighbours_right,
            'neighbours_top': neighbours_top
        })

    return results