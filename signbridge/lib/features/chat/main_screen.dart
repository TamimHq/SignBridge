import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/models.dart';
import '../../shared/theme/app_theme.dart';
import '../avatar/avatar_widget.dart';
import 'chat_controller.dart';
import 'chat_panel.dart';
import 'input_bar.dart';
import '../../signer/signer_panel.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Main Screen
// Layout: top bar | [signer panel | [avatar + chat + input]]
// ─────────────────────────────────────────────────────────────────────────────

class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isWide = MediaQuery.of(context).size.width > 700;

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          children: [
            // ── Top bar ─────────────────────────────────────────────────────
            const _TopBar(),
            const Divider(height: 1, color: AppColors.border),
        
            // ── Main body ───────────────────────────────────────────────────
            Expanded(child: isWide ? const _WideLayout() : const _NarrowLayout()),
          ],
        ),
      ),
    );
  }
}

// ── Top bar ───────────────────────────────────────────────────────────────────
class _TopBar extends StatelessWidget {
  const _TopBar();

  @override
  Widget build(BuildContext context) {
    final chat = context.watch<ChatController>();

    return Container(
      height: 52,
      color: AppColors.surface,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: [
          // Brand
          Container(
            width: 8,
            height: 8,
            decoration: const BoxDecoration(
              color: AppColors.green,
              shape: BoxShape.circle,
              boxShadow: [BoxShadow(color: AppColors.green, blurRadius: 6)],
            ),
          ),
          const SizedBox(width: 10),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'SignBridge',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
              Text(
                'ASL · BdSL · EN · বাংলা',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),

          const Spacer(),

          // Language pills
          _LangPill(
            label: 'ASL / EN',
            active: chat.currentLanguage == AppLanguage.aslEn,
            onTap: () => chat.setLanguage(AppLanguage.aslEn),
          ),
          const SizedBox(width: 6),
          _LangPill(
            label: 'BdSL / বাংলা',
            active: chat.currentLanguage == AppLanguage.bdslBn,
            onTap: () => chat.setLanguage(AppLanguage.bdslBn),
          ),
        ],
      ),
    );
  }
}

class _LangPill extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;
  const _LangPill({
    required this.label,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: active ? AppColors.accent : AppColors.surface2,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: active ? AppColors.accent : AppColors.border,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            color: active ? Colors.white : AppColors.textSecond,
            fontWeight: active ? FontWeight.w500 : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

// ── Wide layout (tablet/desktop) ──────────────────────────────────────────────
class _WideLayout extends StatelessWidget {
  const _WideLayout();

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Left: Signer camera panel
        Container(
          width: 280,
          decoration: const BoxDecoration(
            color: Color(0xFF0D1F3C),
            border: Border(right: BorderSide(color: AppColors.border)),
          ),
          child: const SignerPanel(),
        ),

        // Right: Avatar + Chat + Input
        Expanded(
          child: Column(
            children: [
              // Avatar area (fixed height)
              Container(
                height: 190,
                decoration: const BoxDecoration(
                  color: AppColors.surface,
                  border: Border(bottom: BorderSide(color: AppColors.border)),
                ),
                child: const AvatarWidget(),
              ),
              // Chat messages
              const Expanded(child: ChatPanel()),
              // Input bar
              const Divider(height: 1, color: AppColors.border),
              const InputBar(),
            ],
          ),
        ),
      ],
    );
  }
}

// ── Narrow layout (phone) ─────────────────────────────────────────────────────
class _NarrowLayout extends StatelessWidget {
  const _NarrowLayout();

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Column(
        children: [
          const TabBar(
            tabs: [
              Tab(text: 'Chat'),
              Tab(text: 'Signer'),
            ],
            labelColor: AppColors.accent,
            unselectedLabelColor: AppColors.textSecond,
            indicatorColor: AppColors.accent,
          ),
          Expanded(
            child: TabBarView(
              children: [
                // Chat tab
                Column(
                  children: [
                    Container(
                      height: 160,
                      decoration: const BoxDecoration(
                        color: AppColors.surface,
                        border: Border(
                          bottom: BorderSide(color: AppColors.border),
                        ),
                      ),
                      child: const AvatarWidget(),
                    ),
                    const Expanded(child: ChatPanel()),
                    const Divider(height: 1, color: AppColors.border),
                    const InputBar(),
                  ],
                ),
                // Signer tab
                Container(
                  color: const Color(0xFF0D1F3C),
                  child: const SignerPanel(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
