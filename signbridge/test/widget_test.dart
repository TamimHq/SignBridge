// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:signbridge/main.dart';
import 'package:signbridge/core/lookup_engine.dart';
import 'package:signbridge/core/api_client.dart';

void main() {
  testWidgets('SignBridge app builds without crashing', (
    WidgetTester tester,
  ) async {
    final lookupEngine = SignLookupEngine();
    final apiClient = ApiClient();

    await tester.pumpWidget(
      SignBridgeApp(lookupEngine: lookupEngine, apiClient: apiClient),
    );

    // App should render the top bar brand name
    expect(find.text('SignBridge'), findsOneWidget);
  });
}
