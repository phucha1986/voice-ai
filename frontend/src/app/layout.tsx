import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice AI — Giọng nói tiếng Việt OmniVoice",
  description: "Tổng hợp giọng nói tiếng Việt với OmniVoice — nhân bản từ audio tham chiếu hoặc dùng giọng dựng sẵn.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
