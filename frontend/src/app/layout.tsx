import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice AI — Nhân bản giọng nói XTTS",
  description: "Nhân bản giọng nói từ file audio tham chiếu và tạo giọng nói với XTTS v2, hỗ trợ tiếng Việt.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
