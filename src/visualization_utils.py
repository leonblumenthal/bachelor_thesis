from typing import Any, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.perspective import Perspective

from .models import Detection, Vehicle


def overlay_colored_masks(
    frame: np.ndarray,
    detections: List[Detection],
    colors: Any = (0, 0, 1),
) -> np.ndarray:
    """Color mask region for each detection on copied frame."""

    colors = np.array(px.colors.validate_colors(colors, 'tuple')) * 255

    frame = frame.copy()

    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]

        # Make masked regions grayscale and combine with color.
        x, y = detection.anchor
        h, w = detection.mask.shape
        frame[y : y + h, x : x + w][detection.mask] = (
            frame[y : y + h, x : x + w][detection.mask].mean(axis=-1)[..., None]
            + np.array(color)
        ) // 2

    return frame


def create_detections_figure(
    frame: np.ndarray,
    detections: List[Detection],
    mask_colors: Any = (0, 0, 1),
    box_colors: Any = (0, 1, 0),
    box_thickness: float = 1,
    annotation_colors: Any = (1, 1, 1),
    annotation_size: float = 14,
) -> go.Figure:
    """
    Create figure for visualizing detections on an image including:
    - colored masks
    - colored 2D bounding boxes including line thickness
    - colored id annotations including hover info

    To disable mask, boxes, or annotation, set the respective colors attribute to None.
    """

    # Colorize mask regions.
    if mask_colors is not None:
        frame = overlay_colored_masks(frame, detections, mask_colors)

    # Create figure from (colored) frame and remove excessive margin.
    fig = px.imshow(frame)
    fig.update_layout(margin=dict(r=16, l=16, t=16, b=16))

    # Create 2D bounding boxes.
    if box_colors is not None:
        box_colors = px.colors.validate_colors(box_colors, 'rgb')

        for i, detection in enumerate(detections):
            color = box_colors[i % len(box_colors)]

            x0, y0, x1, y1 = detection.box
            fig.add_shape(
                type='rect',
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line_color=color,
                line_width=box_thickness,
            )

    # Add annotations including hover info.
    if annotation_colors is not None:
        annotation_colors = px.colors.validate_colors(annotation_colors, 'rgb')

        xs = []
        ys = []
        texts = []
        colors = []
        customdata = []

        for i, detection in enumerate(detections):
            x, y = detection.anchor
            h, w = detection.mask.shape
            xs.append(x + w / 2)
            ys.append(y + h / 2)

            texts.append(str(i))
            colors.append(annotation_colors[i % len(annotation_colors)])
            customdata.append((i, detection.label, detection.score))

        fig.add_scatter(
            x=xs,
            y=ys,
            text=texts,
            mode='text',
            textfont=dict(color=colors, size=annotation_size),
            name='',
            customdata=customdata,
            hovertemplate='id: %{customdata[0]}<br>'
            + 'label: %{customdata[1]}<br>score: %{customdata[2]:.2f}',
        )

    return fig


def draw_path(
    fig: go.Figure,
    points: np.ndarray,
    color: Any = (0, 0, 0),
    thickness: float = 2,
    closed: bool = True,
) -> go.Figure:
    """Draw (closed) path on figure with points (2xn)."""

    color = px.colors.validate_colors(color, 'rgb')[0]

    pairs = [f'{x},{y}' for x, y in points.T]
    path = 'M ' + ' L '.join(pairs)
    if closed:
        path += ' Z'

    fig.add_shape(type='path', path=path, line_color=color, line_width=thickness)

    return fig


def draw_contours(
    fig: go.Figure,
    contours: List[np.ndarray],
    colors: Any = (1, 0, 0),
    marker_symbol: str = 'x',
    marker_size: float = 4,
) -> go.Figure:
    """Draw contours (mx2xn) on figure."""

    colors = px.colors.validate_colors(colors, 'rgb')

    for i, points in enumerate(contours):
        color = colors[i % len(colors)]
        fig.add_scatter(
            x=points[0],
            y=points[1],
            mode='markers',
            marker_color=color,
            marker_symbol=marker_symbol,
            marker_size=marker_size,
            name=f'contour {i}',
        )

    return fig


def draw_vehicles(
    fig: go.Figure,
    vehicles: List[Vehicle],
    perspective: Perspective = None,
    colors: Any = (0, 0, 0),
    thickness: float = 2,
) -> go.Figure:
    """
    Draw vehicles as 2D rectangles on ground plane
    or as 3D box on frame if perspective is given.
    """

    colors = px.colors.validate_colors(colors, 'rgb')

    for i, vehicle in enumerate(vehicles):
        color = colors[i % len(colors)]

        # Create local bounding box.
        length, width, height = vehicle.dimensions
        bottom_corners = np.array(
            [
                [-length / 2, +length / 2, +length / 2, -length / 2],
                [-width / 2, -width / 2, +width / 2, +width / 2],
                [0, 0, 0, 0],
            ]
        )
        # Rotate bounding box.
        yaw = vehicle.yaw
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        bottom_corners = rotation @ bottom_corners
        # Add global location
        bottom_corners += vehicle.location

        if perspective is None:
            # Draw bottom corners on ground plane.
            draw_path(fig, bottom_corners[:2], color, thickness)
        else:
            # Add top corners and project onto image.
            height_offset = np.array([[0, 0, height]]).T
            corners = np.hstack((bottom_corners, bottom_corners + height_offset))
            projected_corners = perspective.project_to_image(corners)
            # Draw bottom-, top polygons and vertical lines individually.
            draw_path(fig, projected_corners[:, :4], color, thickness)
            draw_path(fig, projected_corners[:, 4:], color, thickness)
            for j in range(4):
                draw_path(fig, projected_corners[:, [j, j + 4]], color, thickness)

    return fig