import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Converts a camera stream frame (YUV420 on Android, BGRA8888 on iOS) to an RGB image,
/// rotated so it matches what the user sees in the preview.
img.Image cameraImageToImage(CameraImage frame, {int rotationDegrees = 0}) {
  late img.Image out;
  if (frame.format.group == ImageFormatGroup.bgra8888) {
    final p = frame.planes.first;
    out = img.Image.fromBytes(
      width: frame.width,
      height: frame.height,
      bytes: p.bytes.buffer,
      bytesOffset: p.bytes.offsetInBytes,
      rowStride: p.bytesPerRow,
      numChannels: 4,
      order: img.ChannelOrder.bgra,
    );
  } else {
    out = _yuv420ToImage(frame);
  }
  return rotationDegrees == 0 ? out : img.copyRotate(out, angle: rotationDegrees);
}

img.Image _yuv420ToImage(CameraImage frame) {
  final w = frame.width, h = frame.height;
  final yPlane = frame.planes[0], uPlane = frame.planes[1], vPlane = frame.planes[2];
  final uvRowStride = uPlane.bytesPerRow;
  final uvPixelStride = uPlane.bytesPerPixel ?? 1;
  final out = img.Image(width: w, height: h);
  for (var y = 0; y < h; y++) {
    final yRow = y * yPlane.bytesPerRow;
    final uvRow = (y >> 1) * uvRowStride;
    for (var x = 0; x < w; x++) {
      final uvIndex = uvRow + (x >> 1) * uvPixelStride;
      final yy = yPlane.bytes[yRow + x];
      final u = uPlane.bytes[uvIndex] - 128;
      final v = vPlane.bytes[uvIndex] - 128;
      out.setPixelRgb(
        x,
        y,
        (yy + 1.402 * v).round().clamp(0, 255),
        (yy - 0.344136 * u - 0.714136 * v).round().clamp(0, 255),
        (yy + 1.772 * u).round().clamp(0, 255),
      );
    }
  }
  return out;
}

/// Resizes to size x size and returns NHWC float32 RGB values multiplied by [scale].
/// Keras models (MobileNetV3, U-Net) take raw 0-255 (scale 1); YOLO takes 0-1 (scale 1/255).
Float32List imageToInput(img.Image image, int size, {double scale = 1.0}) {
  final resized = image.width == size && image.height == size
      ? image
      : img.copyResize(image, width: size, height: size, interpolation: img.Interpolation.linear);
  final out = Float32List(size * size * 3);
  var i = 0;
  for (final p in resized) {
    out[i++] = p.r * scale;
    out[i++] = p.g * scale;
    out[i++] = p.b * scale;
  }
  return out;
}
