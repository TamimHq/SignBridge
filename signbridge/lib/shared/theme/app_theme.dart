import 'package:flutter/material.dart';

class AppColors {
  static const bg = Color(0xFF0F1117);
  static const surface = Color(0xFF1A1D27);
  static const surface2 = Color(0xFF22263A);
  static const border = Color(0xFF2E3350);
  static const accent = Color(0xFF4F6EF7);
  static const accentPurple = Color(0xFF7C3AED);
  static const green = Color(0xFF22C55E);
  static const red = Color(0xFFEF4444);
  static const textPrimary = Color(0xFFE8EAF6);
  static const textSecond = Color(0xFF8B8FA8);
  static const textThird = Color(0xFF5A5E7A);
  static const signerBg = Color(0xFF0D1F3C);
  static const msgSign = Color(0xFF1A2744);
  static const msgVoice = Color(0xFF1A2732);
}

class AppTheme {
  static ThemeData get dark => ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: AppColors.bg,
    colorScheme: const ColorScheme.dark(
      primary: AppColors.accent,
      secondary: AppColors.accentPurple,
      surface: AppColors.surface,
      error: AppColors.red,
    ),
    fontFamily: 'NotoSansBengali',
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.surface,
      elevation: 0,
      titleTextStyle: TextStyle(
        color: AppColors.textPrimary,
        fontSize: 16,
        fontWeight: FontWeight.w600,
      ),
    ),
    dividerColor: AppColors.border,
    textTheme: const TextTheme(
      bodyLarge: TextStyle(color: AppColors.textPrimary, fontSize: 14),
      bodyMedium: TextStyle(color: AppColors.textPrimary, fontSize: 13),
      bodySmall: TextStyle(color: AppColors.textSecond, fontSize: 11),
      labelSmall: TextStyle(
        color: AppColors.textThird,
        fontSize: 10,
        letterSpacing: 0.8,
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.surface2,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: AppColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: AppColors.accent, width: 1.5),
      ),
      hintStyle: const TextStyle(color: AppColors.textThird, fontSize: 13),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
    ),
  );
}
