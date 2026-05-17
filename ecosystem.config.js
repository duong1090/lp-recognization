module.exports = {
  apps: [
    {
      name: 'lp-service',
      script: 'docker',
      args: [
        'run', '--rm',
        '--name', 'lp-service',
        '--runtime', 'nvidia',       // GPU access on Jetson
        '-p', '8000:8000',
        '-v', '/opt/lp/service/model:/opt/lp/service/model:ro',  // mount weights
        'lp-service:latest',
      ].join(' '),
      interpreter: 'none',
      // Restart policies
      max_restarts: 10,
      restart_delay: 5000,
      kill_timeout: 10000,
      // Stop: PM2 sends SIGINT but we need to stop the container
      kill_retry_time: 3000,
      // Logging
      error_file: '/var/log/lp-service/error.log',
      out_file: '/var/log/lp-service/out.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    },
  ],
};
