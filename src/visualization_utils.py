from typing import Any, List

import cv2
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


def overlay_colored_boxes(
    frame: np.ndarray,
    detections: List[Detection],
    colors: Any = (0, 0, 1),
    thickness: float = 2,
) -> np.ndarray:
    """Overlay colored box for each detection on copied frame."""

    colors = np.array(px.colors.validate_colors(colors, 'tuple'), dtype=float) * 255

    frame = frame.copy()

    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]

        x0, y0, x1, y1 = [int(a) for a in detection.box]

        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)

    return frame


def draw_annotations(
    fig: go.Figure,
    points: np.ndarray,
    texts: List = None,
    colors: Any = (0, 0, 0),
    size: float = 12,
    custom_data: Any = None,
    hover_template: str = None,
) -> go.Figure:
    """
    Draw text annotations at points (2xn) on figure.
    Default text for each point is enumerated index.
    """

    colors = px.colors.validate_colors(colors, 'rgb')
    colors = [colors[i % len(colors)] for i in range(points.shape[1])]

    if texts is None:
        texts = [str(i) for i in range(points.shape[1])]

    x, y = points

    fig.add_scatter(
        x=x,
        y=y,
        text=texts,
        mode='text',
        textfont=dict(color=colors, size=size),
        name='',
        customdata=custom_data,
        hovertemplate=hover_template,
    )

    return fig


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
        points = []
        texts = []
        custom_data = []

        for i, detection in enumerate(detections):
            x, y = detection.anchor
            h, w = detection.mask.shape
            points.append((x + w / 2, y + h / 2))

            texts.append(str(i))
            custom_data.append((i, detection.label, detection.score))

        draw_annotations(
            fig,
            np.array(points).T,
            texts,
            annotation_colors,
            annotation_size,
            custom_data,
            'id: %{customdata[0]}<br>'
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
    marker_colors: Any = (1, 0, 0),
    marker_symbol: str = 'x',
    marker_size: float = 4,
    annotation_colors: Any = None,
    annotation_size: float = 12,
) -> go.Figure:
    """Draw contours (mx2xn) on figure."""

    marker_colors = px.colors.validate_colors(marker_colors, 'rgb')

    for i, points in enumerate(contours):
        color = marker_colors[i % len(marker_colors)]
        fig.add_scatter(
            x=points[0],
            y=points[1],
            mode='markers',
            marker_color=color,
            marker_symbol=marker_symbol,
            marker_size=marker_size,
            name=f'contour {i}',
        )

    if annotation_colors is not None:
        annotation_colors = px.colors.validate_colors(annotation_colors, 'rgb')

        points = np.array([points[:2].mean(1) for points in contours]).T
        draw_annotations(fig, points, colors=annotation_colors, size=annotation_size)

    return fig


def draw_vehicles(
    fig: go.Figure,
    vehicles: List[Vehicle],
    perspective: Perspective = None,
    box_colors: Any = (0, 0, 0),
    box_thickness: float = 2,
    annotation_colors: Any = None,
    annotation_size: float = 12,
) -> go.Figure:
    """
    Draw vehicles as 2D rectangles on ground plane
    or as 3D box on frame if perspective is given.
    """

    box_colors = px.colors.validate_colors(box_colors, 'rgb')

    for i, vehicle in enumerate(vehicles):
        color = box_colors[i % len(box_colors)]

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
            draw_path(fig, bottom_corners[:2], color, box_thickness)
            # Draw direction.
            direction = np.hstack(
                (
                    vehicle.location[:2],
                    bottom_corners[:2, [1, 2]].mean(1, keepdims=True),
                )
            )
            draw_path(fig, direction, color, box_thickness)
        else:
            # Add top corners and project onto image.
            height_offset = np.array([[0, 0, height]]).T
            corners = np.hstack((bottom_corners, bottom_corners + height_offset))
            projected_corners = perspective.project_to_image(corners)
            # Draw bottom-, top polygons and vertical lines individually.
            draw_path(fig, projected_corners[:, :4], color, box_thickness)
            draw_path(fig, projected_corners[:, 4:], color, box_thickness)
            for j in range(4):
                draw_path(fig, projected_corners[:, [j, j + 4]], color, box_thickness)
            # Draw direction.
            direction = np.hstack(
                (
                    projected_corners[:, [0, 2]].mean(1, keepdims=True),
                    projected_corners[:, [1, 2]].mean(1, keepdims=True),
                )
            )
            draw_path(fig, direction, color, box_thickness)

    if annotation_colors is not None:
        annotation_colors = px.colors.validate_colors(annotation_colors, 'rgb')

        if perspective is None:
            # Draw annotation on vehicle locations on ground plane.
            points = np.array([vehicle.location[:2, 0] for vehicle in vehicles]).T
        else:
            # Draw annotations on projected box centers in image.
            points = np.hstack(
                [
                    vehicle.location + np.array([[0, 0, vehicle.dimensions[2] / 2]]).T
                    for vehicle in vehicles
                ]
            )
            points = perspective.project_to_image(points)

        custom_data = [(i, vehicle.category) for i, vehicle in enumerate(vehicles)]

        draw_annotations(
            fig,
            points,
            colors=annotation_colors,
            size=annotation_size,
            custom_data=custom_data,
            hover_template='id: %{customdata[0]}<br>category: %{customdata[1]}',
        )

    return fig


def overlay_3d_vehicles(
    frame: np.ndarray,
    vehicles: List[Vehicle],
    perspective: Perspective = None,
    box_colors: Any = (0, 0, 0),
    box_thickness: float = 2,
) -> go.Figure:
    """
    Overlay vehicles as 3D box on frame.
    """

    frame = frame.copy()

    box_colors = (
        np.array(px.colors.validate_colors(box_colors, 'tuple'), dtype=float) * 255
    )

    for i, vehicle in enumerate(vehicles):
        color = box_colors[i % len(box_colors)]

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

        # Add top corners and project onto image.
        height_offset = np.array([[0, 0, height]]).T
        corners = np.hstack((bottom_corners, bottom_corners + height_offset))
        projected_corners = perspective.project_to_image(corners).astype(int).T

        polygons = []

        # Add bottom-, top polygons and vertical lines individually.
        polygons.append(projected_corners[:4])
        polygons.append(projected_corners[4:])
        for j in range(4):
            polygons.append(projected_corners[[j, j + 4]])
        # Add direction line.
        direction = np.vstack(
            (
                projected_corners[[0, 2]].mean(0),
                projected_corners[[1, 2]].mean(0),
            )
        ).astype(int)

        polygons.append(direction)

        # Draw polygons.
        cv2.polylines(frame, polygons, True, color, box_thickness)

    return frame


def draw_camera_position(
    fig: go.Figure,
    perspective: Perspective,
    color: Any = (0, 0, 0),
) -> go.Figure:
    """Draw camera position of perspective onto ground plane."""

    color = px.colors.validate_colors(color, 'rgb')[0]

    x, y = perspective.translation[:2]
    fig.add_scatter(
        x=x,
        y=y,
        marker=dict(
            size=16,
            symbol='asterisk',
            line_color=color,
            line_width=4,
        ),
        name='camera',
        mode='markers',
    )

    return fig
