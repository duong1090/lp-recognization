module.exports = {
  apps: [
    {
      name: 'lp-service',
      script: 'docker',
      args: [
        'run', '--rm',
        '--name', 'lp-service',
        '--runtime', 'nvidia',
        '-p', '8000:8000',
        'lp-service:latest',
      ].join(' '),
      interpreter: 'none',
      // Restart policies
      max_restarts: 10,
      restart_delay: 5000,
      kill_timeout: 10000,
      // Logging
      error_file: '/var/log/lp-service/error.log',
      out_file: '/var/log/lp-service/out.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    },
  ],
};
