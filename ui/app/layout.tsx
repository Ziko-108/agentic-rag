import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Annual Reports Assistant",
  description: "Cited answers from company annual reports",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
