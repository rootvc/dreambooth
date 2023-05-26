/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "rootvc-photobooth.s3.us-west-2.amazonaws.com",
        port: "",
      },
    ],
  },
};

module.exports = nextConfig;
