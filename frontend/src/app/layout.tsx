import './globals.css'

export const metadata = {
  title: 'Next Click Predictor',
  description: 'AI-powered next click prediction using computer vision and Bayesian networks',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}