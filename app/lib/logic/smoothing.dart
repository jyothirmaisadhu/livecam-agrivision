import 'dart:collection';

/// Majority vote over the last [window] predictions to stop labels flickering in live video.
class MajorityVote {
  MajorityVote({this.window = 5});

  final int window;
  final Queue<String?> _history = Queue();

  String? update(String? label) {
    _history.addLast(label);
    while (_history.length > window) {
      _history.removeFirst();
    }
    final counts = <String, int>{};
    for (final l in _history) {
      if (l != null) counts[l] = (counts[l] ?? 0) + 1;
    }
    if (counts.isEmpty) return null;
    final best = counts.entries.reduce((a, b) => b.value > a.value ? b : a);
    return best.value * 2 > _history.length ? best.key : null;
  }

  void reset() => _history.clear();
}
