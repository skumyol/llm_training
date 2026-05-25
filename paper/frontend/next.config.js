/** @type {import('next').NextConfig} */
const API_HOST = process.env.API_HOST || '127.0.0.1';
const API_PORT = process.env.API_PORT || '8000';

const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `http://${API_HOST}:${API_PORT}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
