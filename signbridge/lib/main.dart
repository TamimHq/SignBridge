import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'core/api_client.dart';
import 'core/lookup_engine.dart';
import 'features/avatar/avatar_controller.dart';
import 'features/chat/chat_controller.dart';
import 'features/chat/main_screen.dart';
import 'shared/theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Pre-load lookup engine indexes
  final lookupEngine = SignLookupEngine();
  await lookupEngine.loadIndexes();

  // Init API client
  final apiClient = ApiClient();
  await apiClient.init();

  runApp(SignBridgeApp(lookupEngine: lookupEngine, apiClient: apiClient));
}

class SignBridgeApp extends StatelessWidget {
  final SignLookupEngine lookupEngine;
  final ApiClient apiClient;

  const SignBridgeApp({
    super.key,
    required this.lookupEngine,
    required this.apiClient,
  });

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // API client (server connectivity)
        ChangeNotifierProvider.value(value: apiClient),

        // Avatar playback controller
        ChangeNotifierProvider(create: (_) => AvatarController()),

        // Chat controller (depends on lookup engine + avatar controller)
        ChangeNotifierProxyProvider<AvatarController, ChatController>(
          create: (ctx) => ChatController(
            lookupEngine: lookupEngine,
            avatarController: Provider.of<AvatarController>(ctx, listen: false),
          ),
          update: (_, avatar, prev) => prev!,
        ),
      ],
      child: MaterialApp(
        title: 'SignBridge',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.dark,
        home: const MainScreen(),
      ),
    );
  }
}
