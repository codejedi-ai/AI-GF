'use client'

import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { HeartIcon, SparklesIcon, ShieldCheckIcon } from '@heroicons/react/24/outline'
import Image from 'next/image'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useUser } from '@auth0/nextjs-auth0/client'

const bannerOverlayStyle = `
  .banner-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to right, rgba(59, 130, 246, 0.7), rgba(168, 85, 247, 0.7));
  }
`;

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()
  const { user, error, isLoading: isUserLoading } = useUser()

  const handleStartSwiping = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/initiated-swipe', { method: 'POST' })
      if (!response.ok) {
        throw new Error('Failed to generate session ID')
      }
      const { sessionId } = await response.json()
      router.push(`/start-swiping?session=${sessionId}`)
    } catch (error) {
      console.error('Error starting swiping session:', error)
      // You might want to show an error message to the user here
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = bannerOverlayStyle;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-ivory-100 via-rose-50 to-earth-100">
      <header className="bg-ivory-200 bg-opacity-90 shadow-md sticky top-0 z-10">
        <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
          <Link href="/" className="flex items-center space-x-2">
            <Image src="/favicon.png" alt="Galatea.AI Logo" width={40} height={40} />
            <span className="text-2xl font-bold text-earth-700">Galatea.AI</span>
          </Link>
          <div className="hidden md:flex space-x-6">
            <Link href="/about" className="text-earth-600 hover:text-rose-700 transition-colors">About</Link>
            <Link href="/profile-setup" className="text-earth-600 hover:text-rose-700 transition-colors">Profile</Link>
          </div>
          <div className="flex space-x-2">
            {!user ? (
              <>
                {/* Log In Button */}
                <Link href="/api/auth/login">
                  <Button variant="ghost" className="text-earth-700 hover:text-rose-700">Log In</Button>
                </Link>

                {/* Sign Up Button */}
                <Link href="/api/auth/login?screen_hint=signup">
                  <Button className="bg-rose-600 text-ivory-100 hover:bg-rose-700">Sign Up</Button>
                </Link>
              </>
            ) : (
              <>
                {/* User Info */}
                <div className="text-earth-700 font-medium">{`Hi, ${user.name}`}</div>

                {/* Log Out Button */}
                <Link href="/api/auth/logout">
                  <Button variant="ghost" className="text-earth-700 hover:text-red-500">Log Out</Button>
                </Link>
              </>
            )}
          </div>
        </nav>
      </header>

      <main className="container mx-auto px-6 py-16">
        <section className="relative h-[500px] mb-20 rounded-xl overflow-hidden">
          <Image
            src="/mekkana-banner.png"
            alt="Mekkana Banner"
            layout="fill"
            objectFit="cover"
            priority
          />
          <div className="banner-overlay"></div>
          <div className="absolute inset-0 flex flex-col justify-center items-center text-center p-6">
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              Sculpt Your Perfect <span className="text-rose-400">AI Companion</span>
            </h1>
            <p className="text-xl md:text-2xl text-ivory-100 mb-10 max-w-3xl">
              Galatea.AI brings the Pygmalion myth to life with cutting-edge artificial intelligence.
            </p>
            {!user ? (
              <p className="text-lg text-white">Please log in or sign up to start swiping!</p>
            ) : (
              <Button 
                onClick={handleStartSwiping} 
                disabled={isLoading}
                size="lg" 
                className="bg-rose-600 text-ivory-100 hover:bg-rose-700 text-xl py-6 px-10"
              >
                {isLoading ? 'Loading...' : 'Start Swiping'}
              </Button>
            )}
          </div>
        </section>

        {/* Features Section */}
        <section className="grid md:grid-cols-3 gap-8 mb-20">
          <FeatureCard 
            icon={<HeartIcon className="h-12 w-12 text-rose-500" />}
            title="Artistic Creation"
            description="Sculpt your ideal AI companion with our advanced personality customization tools."
          />
          <FeatureCard 
            icon={<SparklesIcon className="h-12 w-12 text-rose-500" />}
            title="Bring to Life"
            description="Watch your creation come to life with AI-powered conversations and interactions."
          />
          <FeatureCard 
            icon={<ShieldCheckIcon className="h-12 w-12 text-rose-500" />}
            title="Eternal Devotion"
            description="Experience unwavering companionship and support from your AI partner."
          />
        </section>

        {/* Footer Section */}
        <footer className="bg-earth-100 mt-auto">
          <div className="container mx-auto px-6 py-[10px] flex justify-between items-center">
            © Galatea.AI | All Rights Reserved
          </div>
        </footer>
      </main>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-[20px]">{icon}</div>
  )
}
