import { HTMLAttributes, forwardRef } from 'react'
import { motion, HTMLMotionProps } from 'framer-motion'

interface CardProps extends HTMLMotionProps<"div"> {
  variant?: 'default' | 'elevated' | 'outlined'
  noPadding?: boolean
}

const Card = forwardRef<HTMLDivElement, CardProps>(({
  variant = 'default',
  noPadding = false,
  className = '',
  children,
  ...props
}, ref) => {
  const variants = {
    default: 'glass-card border-cyber-primary/20',
    elevated: 'glass-card border-cyber-primary/30 shadow-lg shadow-cyber-primary/10',
    outlined: 'bg-transparent border-2 border-cyber-primary/40',
  }
  
  return (
    <motion.div
      ref={ref}
      className={`
        rounded-xl border transition-all
        ${variants[variant]}
        ${noPadding ? '' : 'p-4'}
        ${className}
      `}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      {...props}
    >
      {children}
    </motion.div>
  )
})

Card.displayName = 'Card'

// Card sub-components
interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {}

const CardHeader = forwardRef<HTMLDivElement, CardHeaderProps>(({
  className = '',
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={`pb-3 border-b border-gray-700 mb-3 ${className}`}
    {...props}
  >
    {children}
  </div>
))

CardHeader.displayName = 'CardHeader'

interface CardTitleProps extends HTMLAttributes<HTMLHeadingElement> {}

const CardTitle = forwardRef<HTMLHeadingElement, CardTitleProps>(({
  className = '',
  children,
  ...props
}, ref) => (
  <h3
    ref={ref}
    className={`text-lg font-bold text-white font-mono ${className}`}
    {...props}
  >
    {children}
  </h3>
))

CardTitle.displayName = 'CardTitle'

interface CardContentProps extends HTMLAttributes<HTMLDivElement> {}

const CardContent = forwardRef<HTMLDivElement, CardContentProps>(({
  className = '',
  children,
  ...props
}, ref) => (
  <div ref={ref} className={className} {...props}>
    {children}
  </div>
))

CardContent.displayName = 'CardContent'

interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {}

const CardFooter = forwardRef<HTMLDivElement, CardFooterProps>(({
  className = '',
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={`pt-3 border-t border-gray-700 mt-3 ${className}`}
    {...props}
  >
    {children}
  </div>
))

CardFooter.displayName = 'CardFooter'

export { Card, CardHeader, CardTitle, CardContent, CardFooter }
export default Card
