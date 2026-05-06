# Windows ECS Operations Guide

这份文档只负责日常运维，不重复部署细节。

适用目录：

- 后端源码：`D:\ashare-quant\app\backend`
- 前端源码：`D:\ashare-quant\app\frontend`
- 前端发布目录：`D:\wwwroot\lianghua-web`

常用访问地址：

- 前端：`http://116.62.21.194:8080/overview`
- 信号页：`http://116.62.21.194:8080/signals`
- 后端健康检查：`http://116.62.21.194:8000/health`

## 常用脚本

脚本都在：

- `D:\ashare-quant\app\backend\deploy\windows\scripts`

最常用的 4 个：

- `restart_backend.ps1`
- `update_backend.ps1`
- `update_frontend.ps1`
- `install_refresh_task.ps1`

## 后端更新

普通后端代码更新：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1
```

如果改了 `requirements.txt`：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1 -InstallDeps
```

如果改了同步、训练、模型逻辑，并且要立刻产出新结果：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1 -RunRefresh
```

## 前端更新

普通前端代码更新：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_frontend.ps1 -PublishRoot "D:\wwwroot\lianghua-web"
```

如果改了 `package.json`：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_frontend.ps1 -PublishRoot "D:\wwwroot\lianghua-web" -InstallDeps
```

## 只重启后端

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\restart_backend.ps1
```

## 前后端都更新

推荐顺序：

1. 更新后端
2. 重启后端或重跑 refresh
3. 更新前端
4. 打开页面检查

命令：

```powershell
cd D:\ashare-quant\app\backend
git pull
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1

cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_frontend.ps1 -PublishRoot "D:\wwwroot\lianghua-web"
```

如果后端改动涉及训练结果：

```powershell
cd D:\ashare-quant\app\backend
powershell -ExecutionPolicy Bypass -File .\deploy\windows\scripts\update_backend.ps1 -RunRefresh
```

## 每日自动任务

后端常驻任务：

```powershell
Get-ScheduledTask -TaskName AshareQuantBackend
Get-ScheduledTaskInfo -TaskName AshareQuantBackend
```

每日刷新任务：

```powershell
Get-ScheduledTask -TaskName AshareQuantDailyRefresh
Get-ScheduledTaskInfo -TaskName AshareQuantDailyRefresh
```

手动触发每日刷新：

```powershell
Start-ScheduledTask -TaskName AshareQuantDailyRefresh
```

## 日志位置

后端服务日志：

- `D:\ashare-quant\logs\backend.out.log`
- `D:\ashare-quant\logs\backend.err.log`

每日刷新日志：

- `D:\ashare-quant\logs\refresh.out.log`
- `D:\ashare-quant\logs\refresh.err.log`

查看最近日志：

```powershell
Get-Content D:\ashare-quant\logs\backend.out.log -Tail 50
Get-Content D:\ashare-quant\logs\backend.err.log -Tail 50
Get-Content D:\ashare-quant\logs\refresh.out.log -Tail 50
Get-Content D:\ashare-quant\logs\refresh.err.log -Tail 50
```

## 更新后检查

后端检查：

- `http://116.62.21.194:8000/health`

前端检查：

- `http://116.62.21.194:8080/overview`
- `http://116.62.21.194:8080/signals`

## 常见情况

改了这些内容时，需要额外注意：

- 改了 `.env`：重启后端
- 改了 `frontend/.env.production`：重新构建前端
- 改了 `requirements.txt`：重新安装 Python 依赖
- 改了 `package.json`：重新安装前端依赖
- 改了模型、因子、训练、同步逻辑：执行 `-RunRefresh`
