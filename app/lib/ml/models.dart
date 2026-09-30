import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../logic/engine.dart';
import '../logic/yolo.dart';
import 'preprocess.dart';

const modelDir = 'assets/models';

class TfliteModel {
  TfliteModel._(this.interpreter) {
    interpreter.allocateTensors();
    inputSize = interpreter.getInputTensor(0).shape[1];
    outputShape = interpreter.getOutputTensor(0).shape;
  }

  final Interpreter interpreter;
  late final int inputSize;
  late final List<int> outputShape;

  /// Returns null when the asset is missing so the app runs with whichever models are bundled.
  static Future<TfliteModel?> tryLoad(String asset) async {
    try {
      final options = InterpreterOptions()..threads = 2;
      return TfliteModel._(await Interpreter.fromAsset(asset, options: options));
    } catch (_) {
      return null;
    }
  }

  Float32List run(Float32List input) {
    interpreter.getInputTensor(0).setTo(input.buffer.asUint8List(input.offsetInBytes, input.lengthInBytes));
    interpreter.invoke();
    var bytes = interpreter.getOutputTensor(0).data;
    if (bytes.offsetInBytes % 4 != 0) bytes = Uint8List.fromList(bytes);
    return bytes.buffer.asFloat32List(bytes.offsetInBytes, bytes.lengthInBytes ~/ 4);
  }

  void close() => interpreter.close();
}

class DiseaseModel {
  DiseaseModel(this.model, this.labels);

  final TfliteModel model;
  final List<String> labels;

  DiseaseResult classify(img.Image image) {
    final probs = model.run(imageToInput(image, model.inputSize));
    var best = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) best = i;
    }
    return DiseaseResult(labels[best], probs[best]);
  }
}

class SegMask {
  const SegMask(this.values, this.size);

  /// size x size values: 0 background, 1 leaf, 2 lesion.
  final Uint8List values;
  final int size;
}

class DamageModel {
  DamageModel(this.model);

  final TfliteModel model;

  SegMask segment(img.Image image) {
    final size = model.inputSize;
    final probs = model.run(imageToInput(image, size));
    final classes = model.outputShape.last;
    final mask = Uint8List(size * size);
    for (var i = 0; i < mask.length; i++) {
      var best = 0;
      for (var c = 1; c < classes; c++) {
        if (probs[i * classes + c] > probs[i * classes + best]) best = c;
      }
      mask[i] = best;
    }
    return SegMask(mask, size);
  }
}

class PestModel {
  PestModel(this.model, this.labels, this.confidence);

  final TfliteModel model;
  final List<String> labels;
  final double confidence;

  List<PestDetection> detect(img.Image image) {
    final out = model.run(imageToInput(image, model.inputSize, scale: 1 / 255));
    return decodeYolo(out, labels,
        numAnchors: model.outputShape.last, inputSize: model.inputSize, confidence: confidence);
  }
}

class AgriModels {
  AgriModels({this.disease, this.pest, this.damage});

  final DiseaseModel? disease;
  final PestModel? pest;
  final DamageModel? damage;

  static Future<AgriModels> load({required double pestConfidence}) async {
    final disease = await TfliteModel.tryLoad('$modelDir/disease_mobilenetv3.tflite');
    final damage = await TfliteModel.tryLoad('$modelDir/damage_unet.tflite');
    final pest = await TfliteModel.tryLoad('$modelDir/pest_yolo11n.tflite');
    return AgriModels(
      disease: disease == null
          ? null
          : DiseaseModel(
              disease,
              (jsonDecode(await rootBundle.loadString('$modelDir/disease_labels.json')) as List)
                  .cast<String>()),
      damage: damage == null ? null : DamageModel(damage),
      pest: pest == null
          ? null
          : PestModel(
              pest,
              const LineSplitter()
                  .convert(await rootBundle.loadString('$modelDir/pest_labels.txt'))
                  .where((l) => l.trim().isNotEmpty)
                  .toList(),
              pestConfidence,
            ),
    );
  }
}
